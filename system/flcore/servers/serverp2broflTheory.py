import time
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import copy
from threading import Thread


class P2BroFLTheory(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.picked_krum = args.picked_krum

        self.set_slow_clients()
        self.set_clients(clientAVG)

        self.model_snapshot = copy.deepcopy(self.global_model)

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
            self.defense_p2brofl_theory()
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

    def defense_p2brofl_theory(self):
        assert (len(self.uploaded_models) > 0)

        len_uploaded_models = len(self.uploaded_models)

        cosine_matrix = [
            [
                self.cosine_similarity(
                    self.vector_norm(self.get_vector_from_params(self.uploaded_models[i])),
                    self.vector_norm(self.get_vector_from_params(self.uploaded_models[j]))
                )
                for j in range(len_uploaded_models)
            ]
            for i in range(len_uploaded_models)
        ]

        num_clients_to_pick = int(self.picked_krum * len_uploaded_models)
        client_scores = [
            np.sum(cosine_matrix[i])
            for i in range(len_uploaded_models)
        ]

        picked_clients_indices = np.argpartition(client_scores, -num_clients_to_pick)[-num_clients_to_pick:]

        tmp_uploaded_ids = []
        tmp_uploaded_weights = []
        tmp_uploaded_models = []
        tot_weights = 0.0
        for i in picked_clients_indices:
            tot_weights += self.uploaded_weights[i]
            tmp_uploaded_ids.append(self.uploaded_ids[i])
            tmp_uploaded_weights.append(self.uploaded_weights[i])
            tmp_uploaded_models.append(self.uploaded_models[i])
        for i, w in enumerate(tmp_uploaded_weights):
            tmp_uploaded_weights[i] = w / tot_weights

        self.uploaded_ids = tmp_uploaded_ids
        self.uploaded_weights = tmp_uploaded_weights
        self.uploaded_models = tmp_uploaded_models
