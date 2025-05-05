import copy
import time
import json
import random
import numpy as np

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from utils.encryption import *


class ProFLCipher(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.lhe_public_key = load_public_key_safely("./flcore/LHE_KEYS/lhe_key_pairs.json")
        self.lhe_enc_func = np.vectorize(self.lhe_public_key.encrypt)
        self.server_SP = SP()

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

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

            self.defense_proFL_cipher()
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

                # virtual encryption channel
                client_params = self.get_vector_from_params(client.model)
                client_params_enc = self.lhe_enc_func(client_params)

                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client_params_enc)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        len_uploaded_models = len(self.uploaded_models)
        len_params = len(self.uploaded_models[0])

        global_vector_params_enc = np.zeros((len_params,)).astype(paillier.EncryptedNumber)
        for i in range(len_uploaded_models):
            global_vector_params_enc += self.uploaded_weights[i] * self.uploaded_models[i]

        global_vector_params = self.server_SP.secure_decrypt(global_vector_params_enc)
        self.set_params_from_vector(self.global_model, global_vector_params)

    def defense_proFL_cipher(self):
        assert (len(self.uploaded_models) > 0)

        len_uploaded_models = len(self.uploaded_models)
        len_params = len(self.uploaded_models[0])

        matrix_params_enc = np.array(self.uploaded_models)

        # SSecMed
        vector_med_enc = self.SSecMed(len_params, matrix_params_enc)

        # Distances
        lst_mh_dis = []
        for i in range(len_uploaded_models):
            mh_dis_med = self.SecMHD(matrix_params_enc[i], vector_med_enc)  # SecMHD
            lst_mh_dis.append(mh_dis_med)

        # Weights
        client_scores = []
        max_mh_dis = max(lst_mh_dis)
        for i in range(len_uploaded_models):
            client_scores.append(max_mh_dis - lst_mh_dis[i])

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

    def SSecMed(self, len_params, matrix_params_enc):
        """
        # Shuffling-based Secure Median (SSMed)
        # Correctness has been verified.
        """
        # AGG: encrypted
        R = self.random_perturbation(len_params)  # Random number perturbation (R)
        O = np.random.permutation(len_params)  # random sequence (O)

        matrix_params_perturbed = matrix_params_enc + R
        matrix_params_perturbed_shuffled = matrix_params_perturbed[:, O]

        # SP: decrypted
        vector_med_enc_perturbed_shuffled = self.server_SP.secure_median(matrix_params_perturbed_shuffled)

        # AGG: encrypted
        vector_med_enc_perturbed = np.zeros(len_params, dtype=paillier.EncryptedNumber)
        for j, o in zip(np.arange(len_params), O):
            vector_med_enc_perturbed[o] = vector_med_enc_perturbed_shuffled[j]
        vector_med_enc = vector_med_enc_perturbed - R

        return vector_med_enc

    def SecMHD(self, vec1_enc, vec2_enc):
        """
        # Secure Manhattan Distance (SecMHD)
        # Correctness has been verified.
        """
        # AGG: encrypted
        diff_enc = vec1_enc - vec2_enc
        R = self.random_perturbation(len(vec1_enc))  # Random number perturbation (R)
        diff_enc_R = diff_enc * R

        # SP: decrypted
        diff_R_sign = self.server_SP.secure_sign(diff_enc_R)

        # AGG: encrypted
        mh_dis_enc = 0
        for s, r, d in zip(diff_R_sign, np.sign(R), diff_enc):
            mh_dis_enc += s * r * d

        mh_dis = self.server_SP.secure_decrypt_one(mh_dis_enc)

        return mh_dis

    def random_perturbation(self, dims):
        """
            Generate a random integer vector of dimension dims, where each number is not zero.
        """
        dims1 = int(dims / 2)
        dims2 = dims - dims1
        random_vec1 = np.random.randint(-100, 0, size=(dims1,))
        random_vec2 = np.random.randint(1, 101, size=(dims2,))
        random_vec = np.concatenate((random_vec1, random_vec2))
        np.random.shuffle(random_vec)

        return random_vec


class SP(object):
    def __init__(self):
        self.__lhe_public_key = load_public_key_safely("./flcore/LHE_KEYS/lhe_key_pairs.json")
        self.__lhe_private_key = load_private_key_safely("./flcore/LHE_KEYS/lhe_key_pairs.json")

        self.__lhe_enc_func = np.vectorize(self.__lhe_public_key.encrypt)
        self.__lhe_dec_func = np.vectorize(self.__lhe_private_key.decrypt)

    def secure_median(self, ENC_matrix):
        DEC_matrix = self.__lhe_dec_func(ENC_matrix)
        vector_med = self.median(DEC_matrix)
        ENC_vector_med = self.__lhe_enc_func(vector_med)
        return ENC_vector_med

    def secure_sign(self, ENC_difference):
        DEC_difference = self.__lhe_dec_func(ENC_difference)
        DEC_difference_sign = np.sign(DEC_difference)
        return DEC_difference_sign

    def secure_decrypt(self, ENC_nums):
        DEC_nums = self.__lhe_dec_func(ENC_nums)
        return DEC_nums

    def secure_decrypt_one(self, ENC_num):
        DEC_num = self.__lhe_private_key.decrypt(ENC_num)
        return DEC_num

    def median(self, A):
        num_rows, num_cols = A.shape
        median_pos = int(num_rows / 2)
        median = [
            np.mean(
                np.sort(A[:, col])[median_pos]
            )
            for col in range(num_cols)
        ]
        return np.array(median)
