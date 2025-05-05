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

# !/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverkrum import Krum
from flcore.servers.servermedian import Median
from flcore.servers.servertrimmean import TrimMean
from flcore.servers.serverbulyan import Bulyan
from flcore.servers.serverfltrust import FLTrust
from flcore.servers.servervrfl import VarianceReductionFL
from flcore.servers.serverpeflTheory import PEFLTheory
from flcore.servers.serversearTheory import SEARTheory
from flcore.servers.serverp2broflTheory import P2BroFLTheory
from flcore.servers.servershieldflTheory import ShieldFLTheory
from flcore.servers.serverproflTheory import ProFLTheory
from flcore.servers.serverproflCipher import ProFLCipher


from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32

# hyper-params for dim1rnn model on ids datasets
hidden_size = 200
num_layers = 2


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":  # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" == args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Cifar100" == args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=40000).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            # Use the pre-trained parameters and set the classification categories.
            args.model = torchvision.models.resnet18(pretrained=True)
            num_ftrs = args.model.fc.in_features
            args.model.fc = nn.Linear(num_ftrs, args.num_classes)
            args.model = args.model.to(args.device)

        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                   output_size=args.num_classes, num_layers=1,
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                   embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, nlayers=2,
                                          num_classes=args.num_classes, max_len=max_len).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Krum":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Krum(args, i)

        elif args.algorithm == "Median":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Median(args, i)

        elif args.algorithm == "TrimMean":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = TrimMean(args, i)

        elif args.algorithm == "Bulyan":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Bulyan(args, i)

        elif args.algorithm == "FLTrust":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FLTrust(args, i)

        elif args.algorithm == "VarianceReductionFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = VarianceReductionFL(args, i)

        elif args.algorithm == "PEFLTheory":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PEFLTheory(args, i)

        elif args.algorithm == "SEARTheory":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = SEARTheory(args, i)

        elif args.algorithm == "P2BroFLTheory":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = P2BroFLTheory(args, i)

        elif args.algorithm == "ShieldFLTheory":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = ShieldFLTheory(args, i)

        elif args.algorithm == "ProFLTheory":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = ProFLTheory(args, i)

        elif args.algorithm == "ProFLCipher":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = ProFLCipher(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # Poisoning attack Module
    parser.add_argument('-attr', "--attack_rate", type=float, default=0.0)
    parser.add_argument('-attc', "--attack_category", type=int, default=0)
    # Defense strategy
    parser.add_argument('-is_mkrum', "--is_mkrum", type=bool, default=False)
    parser.add_argument('-picked_krum', "--picked_krum", type=float, default=0.8)
    parser.add_argument('-trim_fraction', "--trim_fraction", type=float, default=0.15)
    parser.add_argument('-roottrain_size', "--roottrain_size", type=int, default=200)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("Category of attacks: {}".format(args.attack_category))
    print("Rate of attacks: {}".format(args.attack_rate))
    print("=" * 50)

    run(args)
