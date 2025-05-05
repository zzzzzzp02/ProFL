import copy
import time
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data


class FLTrust(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.train_size = args.roottrain_size

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.model_snapshot = copy.deepcopy(self.global_model)
        self.root_model = copy.deepcopy(self.global_model)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.defenseFLTrust()
            self.aggregate_parameters()

            self.model_snapshot = copy.deepcopy(self.global_model)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def defenseFLTrust(self):
        assert (len(self.uploaded_models) > 0)

        self.train_root_dataset(max_epochs=self.local_epochs, train_size=self.train_size)

        len_uploaded_models = len(self.uploaded_models)

        vector_snap_shot = self.get_vector_from_params(self.model_snapshot)
        vector_root = self.get_vector_from_params(self.root_model)

        list_vector_models = []
        for i in range(len(self.uploaded_models)):
            list_vector_models.append(self.get_vector_from_params(self.uploaded_models[i]))
        list_vector_models.append(vector_root)

        list_vector_models = [vi - vector_snap_shot for vi in list_vector_models]
        matrix_models = np.array(list_vector_models)

        scaler = StandardScaler()
        matrix_models_scaler = scaler.fit_transform(matrix_models)

        client_scores = []
        for i in range(len_uploaded_models):
            client_scores.append(self.cosine_similarity(matrix_models_scaler[i], matrix_models_scaler[-1]))

        client_scores = [self.relu(x) for x in client_scores]

        if np.sum(client_scores) == 0:
            client_scores = [1 / len(client_scores)] * len(client_scores)

        tmp_uploaded_ids = []
        tmp_uploaded_weights = []
        tmp_uploaded_models = []
        tot_weights = 0.0
        for i in range(len_uploaded_models):
            tot_weights += client_scores[i]
            tmp_uploaded_ids.append(self.uploaded_ids[i])
            tmp_uploaded_weights.append(client_scores[i])
            tmp_uploaded_models.append(self.uploaded_models[i])
        for i, w in enumerate(tmp_uploaded_weights):
            tmp_uploaded_weights[i] = w / tot_weights

        self.uploaded_ids = tmp_uploaded_ids
        self.uploaded_weights = tmp_uploaded_weights
        self.uploaded_models = tmp_uploaded_models

    def train_root_dataset(self, max_epochs, train_size):
        trainloader = self.load_roottrain_data(train_size)
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(self.root_model.parameters(), lr=self.learning_rate)

        self.root_model.train()
        for epoch in range(max_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.root_model(x)
                loss = loss_fn(output, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # print('root model have been perfectly trained on root dataset.')

    def load_roottrain_data(self, train_size, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        all_train_data = []
        for i in range(self.num_clients):
            all_train_data.extend(read_client_data(self.dataset, i, is_train=True))
        roottrain_data = random.sample(all_train_data, train_size)
        return DataLoader(roottrain_data, batch_size, drop_last=True, shuffle=True)
