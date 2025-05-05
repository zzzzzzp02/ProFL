import copy
import time
import numpy as np

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class ProFLTheory(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

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

            self.defense_proFL_theory()
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

    def defense_proFL_theory(self):
        assert (len(self.uploaded_models) > 0)

        len_uploaded_models = len(self.uploaded_models)
        len_params = len(self.get_vector_from_params(self.global_model))

        list_vector_models = []
        for i in range(len(self.uploaded_models)):
            list_vector_models.append(self.get_vector_from_params(self.uploaded_models[i]))
        matrix_models = np.array(list_vector_models)

        # SSecMed Theory
        vector_med = self.SSecMed(len_params, matrix_models)

        # Manhattan Distances
        lst_mh_dis = []
        for i in range(len_uploaded_models):
            mh_dis_med = self.SecMHD(matrix_models[i], vector_med)  # SecMHD Theory
            lst_mh_dis.append(mh_dis_med)

        # Aggregation weights
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

    def SSecMed(self, len_params, matrix_models):
        """
        # Shuffling-based Secure Median (SSMed)
        # Correctness has been verified.
        """
        # AGG: encrypted
        R = self.random_perturbation(len_params)  # Random number perturbation
        O = np.random.permutation(len_params)  # random sequence O

        matrix_models_perturbed = matrix_models + R
        matrix_models_perturbed_shuffled = matrix_models_perturbed[:, O]

        # SP: decrypted
        vector_med_perturbed_shuffled = self.median(matrix_models_perturbed_shuffled)

        # AGG: encrypted
        vector_med_perturbed = np.zeros(len_params)
        for j, o in zip(np.arange(len_params), O):
            vector_med_perturbed[o] = vector_med_perturbed_shuffled[j]
        vector_med = vector_med_perturbed - R

        return vector_med

    def SecMHD(self, vec1, vec2):
        """
        # Secure Manhattan Distance (SecMHD)
        # Correctness has been verified.
        """
        # AGG: encrypted
        diff = vec1 - vec2
        R = self.random_perturbation(len(vec1))  # Random number perturbation
        diff_R = diff * R

        # SP: decrypted
        diff_R_sign = np.sign(diff_R)

        # AGG: encrypted
        mh_dis = 0
        for s, r, d in zip(diff_R_sign, np.sign(R), diff):
            mh_dis += s * r * d

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
