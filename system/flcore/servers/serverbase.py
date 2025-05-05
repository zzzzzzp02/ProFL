# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import math
from collections import OrderedDict

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from utils.attack import *

from collections import defaultdict


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        # select attack clients
        self.attack_rate = args.attack_rate
        self.attack_category = args.attack_category
        self.attack_clients = [0 for _ in range(self.num_clients)]
        self.set_attack_clients()

    # Modified by zzzzzzp
    def set_clients(self, clientObj):
        for i, train_slow, send_slow, is_attacker in zip(range(self.num_clients), self.train_slow_clients,
                                                         self.send_slow_clients, self.attack_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow,
                               is_attacker=is_attacker)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # model poisoning attacks
        self.MPAs()

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc, _ = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    #####################################################################################
    # Added by zzzzzzp

    def set_attack_clients(self):
        idx = np.array([i for i in range(self.num_clients)])
        idx_ = np.random.choice(idx, math.floor(self.attack_rate * self.num_clients), replace=False)
        for i in idx_:
            self.attack_clients[i] = self.attack_category

    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray):
        if vec1.shape != vec2.shape:
            return None
        return np.square(np.linalg.norm(vec1 - vec2, ord=2))

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        similarity = dot_product / (norm_vec1 * norm_vec2)

        return similarity

    # ReLu
    def relu(self, x):
        return np.where(x >= 0, x, 0)

    def pearson_correlation_coefficient(self, vec1: np.ndarray, vec2: np.ndarray):
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        correlation_matrix = np.corrcoef(vec1, vec2)
        pearson_r = correlation_matrix[0, 1]
        return pearson_r

    def geometric_median(self, X, max_iter=100, tol=1e-4):
        """
        Calculate the geometric median of multiple local models.
            num_rows: The number of local models of the participants
            num_cols: The dimension of each model parameters
        """
        x0 = X.mean(axis=0)
        for i in range(max_iter):
            # Weiszfeld's algorithm as a fixed-point iteration
            weights = 1 / np.sqrt(((X - x0) ** 2).sum(axis=1))
            weights /= weights.sum()
            x1 = (X * weights[:, None]).sum(axis=0)

            # Check for convergence
            if np.linalg.norm(x1 - x0) < tol:
                return x1
            x0 = x1

        return x0

    def median(self, A):
        """
        Calculate the Median model of multiple local models.
            num_rows: The number of local models of the participants
            num_cols: The dimension of each model parameters
        """
        num_rows, num_cols = A.shape
        median_pos = int(num_rows / 2)
        median = [
            np.mean(
                np.sort(A[:, col])[median_pos]
            )
            for col in range(num_cols)
        ]
        return np.array(median)

    def trim_mean(self, A, trim_fraction=0.2):
        """
        Calculate the TrimMean model of multiple local models.
            num_rows: The number of local models of the participants
            num_cols: The dimension of each model parameters
        """
        num_rows, num_cols = A.shape
        trimmed_size = int(trim_fraction * num_rows) if int(trim_fraction * num_rows) > 0 else 1
        if trimmed_size >= int(num_rows / 2):
            raise ValueError("trimmed_size >= int(num_rows / 2) -> Unable Trim.")
        trimmed_mean = [
            np.mean(
                np.sort(A[:, col])[trimmed_size: -trimmed_size]
            )
            for col in range(num_cols)
        ]
        return np.array(trimmed_mean)

    # Convert any model parameters to a Numpy vector.
    def get_vector_from_params(self, net) -> np.ndarray:
        lst_ndarray_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        new_vec = np.array([]).reshape(-1)
        for i in range(len(lst_ndarray_params)):
            new_vec = np.concatenate([new_vec, lst_ndarray_params[i].reshape(-1)])
        return new_vec

    # Store any Numpy vector in net in sequence.
    def set_params_from_vector(self, net, vector: np.ndarray):
        format_lst_ndarray_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        new_numpy = []
        for i in range(len(format_lst_ndarray_params)):
            len_layer = format_lst_ndarray_params[i].reshape(-1).shape[0]
            vector_first = np.array(vector[:len_layer])
            vector_without_first = np.array(vector[len_layer:])
            new_numpy.append(vector_first.reshape(format_lst_ndarray_params[i].shape))
            vector = vector_without_first

        params_dict = zip(net.state_dict().keys(), new_numpy)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # Model vector normalization
    def vector_norm(self, model_vector):
        return model_vector / np.linalg.norm(model_vector)

    # model poisoning attacks
    def MPAs(self):
        """
        is_attacker =
            2: Sign-flipping attack [RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets]
            3: Gaussian-noise attack [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning]
            4: Little is enough (LIE) attack [A Little Is Enough: Circumventing Defenses For Distributed Learning]
            5: Fang's attack [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning]
            6: Min-Max attack [Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning]
            7: AGR-tailored attack [Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning]
            8: FMPA [Denial-of-service or fine-grained control: Towards flexible model poisoning attacks on federated learning]
        """
        # malicious clients
        mal_clients = [c for c in self.clients if c.is_attacker]

        if len(mal_clients) == 0:
            return

        if self.attack_category == 1:  # label-flipping attack
            # implemented in <flcore.clients.clientbase.py>
            return

        if self.attack_category == 2:  # sign-flipping attack
            for mal_c in mal_clients:
                vector_model = mal_c.get_vector_from_params(mal_c.model)
                mal_c.set_params_from_vector(mal_c.model, -1.0 * vector_model)

        elif self.attack_category == 3:  # gaussian-noise attack
            for mal_c in mal_clients:
                vector_model = mal_c.get_vector_from_params(mal_c.model)
                gaussian_noise = np.random.normal(loc=0, scale=1, size=len(vector_model))
                mal_c.set_params_from_vector(mal_c.model, gaussian_noise)

        elif self.attack_category == 4:  # LIE attack
            from scipy.stats import norm
            vector_models = [c.get_vector_from_params(c.model) for c in mal_clients]
            n, m = self.num_clients, len(mal_clients)

            mu = np.mean(vector_models, axis=0)
            sigma = np.mean(vector_models, axis=0)
            for mal_c in mal_clients:
                drift_mean = mu - norm.ppf((n / 2 - 1) / (n - m)) * sigma
                mal_c.set_params_from_vector(mal_c.model, drift_mean)

        elif self.attack_category == 5:  # Fang's attack
            agg_type = 'krum'
            if agg_type == 'median':
                benign_updates = torch.tensor(
                    [c.get_vector_from_params(c.model) for c in self.clients if not c.is_attacker])
                mal_updates = attack_median_and_trimmedmean(benign_updates, len(mal_clients))
            elif agg_type == 'krum':
                all_updates = torch.tensor([c.get_vector_from_params(c.model) for c in self.clients])
                mal_updates = get_malicious_updates_fang(all_updates, len(mal_clients))
            elif agg_type == 'adap':
                all_updates = torch.tensor([c.get_vector_from_params(c.model) for c in self.clients])
                mal_updates = fang_adap(all_updates, len(mal_clients))

            for mal_c, mal_u in zip(mal_clients, mal_updates):
                mal_c.set_params_from_vector(mal_c.model, np.array(mal_u))

        elif self.attack_category == 6:  # Min-Max attack
            all_updates = torch.tensor([c.get_vector_from_params(c.model) for c in self.clients])
            mal_updates = min_max(all_updates, len(mal_clients), dev_type='unit_vec')
            for mal_c, mal_u in zip(mal_clients, mal_updates):
                mal_c.set_params_from_vector(mal_c.model, np.array(mal_u))

        elif self.attack_category == 7:  # AGR-tailored attack
            all_updates = torch.tensor([c.get_vector_from_params(c.model) for c in self.clients])

            agg_type = 'krum'
            if agg_type == 'krum':
                mal_updates = AGR_tailored_attack_on_krum(all_updates, len(mal_clients), dev_type='unit_vec')
            elif agg_type == 'median':
                mal_updates = AGR_tailored_attack_on_median(all_updates, len(mal_clients), dev_type='unit_vec')
            elif agg_type == 'trmean':
                mal_updates = AGR_tailored_attack_on_trmean(all_updates, len(mal_clients), dev_type='unit_vec')

            for mal_c, mal_u in zip(mal_clients, mal_updates):
                mal_c.set_params_from_vector(mal_c.model, np.array(mal_u))

        elif self.attack_category == 8:  # FMPA
            benign_updates = [self.get_vector_from_params(c.model) for c in self.clients if not c.is_attacker]
            target_updates = self.get_vector_from_params(self.global_model)
            malicious_update_one = FMPA_malicious(target_updates, len(self.clients), benign_updates, choice=1, p=10)

            for mal_c in mal_clients:
                mal_c.set_params_from_vector(mal_c.model, np.array(malicious_update_one))

        else:
            raise NotImplementedError
