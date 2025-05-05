# Byzantine-robust Federated Learning Considering Distributed Non-IID Data in Privacy-sensitive Scenarios

We will continue to provide detailed introductions and updates on the repository.

***

This project is based on the open source project [PFLlib](https://github.com/TsingZ0/PFLlib) development.

## The datsets used in this project

- MNIST
- FashionMNIST
- Cifar10
- FEMNIST
- Cifar100
- Caltech256

## The types of poisoning attacks

- **Label-flipping attack** — [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang) *USENIX Security 2020*
- **Sign-flipping attack** — [RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets](https://ojs.aaai.org/index.php/AAAI/article/view/3968) *AAAI 2019*
- **Gaussian-noise attack** — [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang) *USENIX Security 2020*
- **LIE attack** — [A little is enough: Circumventing defenses for distributed learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html) *NeurIPS 2019*
- **Fang's attack** — [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang) *USENIX Security 2020*
- **Min-Max attack** — [Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf) *NDSS 2021*
- **AGR-tailored attack** — [Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf) *NDSS 2021*
- **FMPA** — [Denial-of-service or fine-grained control: Towards flexible model poisoning attacks on federated learning](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/508) *IJCAI 2023*


## The types of Robust Federated Learning

 ***Byzantine-robust Federated Learning***

- **Krum/MultiKrum** — [Machine learning with adversaries: Byzantine tolerant gradient descent](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) *NeurIPS 2017*
- **Median/TrimMean** — [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a) *PMLR 2018*
- **Bulyan** — [The Hidden Vulnerability of Distributed Learning in Byzantium](https://proceedings.mlr.press/v80/mhamdi18a.html) *PMLR 2018*
- **FLTrust** — [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf) *NDSS 2021*
- **Variance-reduced FL** — [Byzantine-robust variance-reduced federated learning over distributed non-i.i.d. data](https://www.sciencedirect.com/science/article/pii/S0020025522012464) *Information Sciences 2022*

 ***Privacy-perserving and Byzantine-robust Federated Learning (Theoretical deployment)***

- **PEFL** — [Privacy-Enhanced Federated Learning Against Poisoning Adversaries](https://ieeexplore.ieee.org/abstract/document/9524709) *TIFS 2021*
- **SEAR** — [SEAR: Secure and Efficient Aggregation for Byzantine-Robust Federated Learning](https://ieeexplore.ieee.org/abstract/document/9468910) *TDSC 2022*
- **ShieldFL** — [ShieldFL: Mitigating Model Poisoning Attacks in Privacy-Preserving Federated Learning](https://ieeexplore.ieee.org/abstract/document/9762272) *TIFS 2022*
- **P2BroFL** — [Privacy-Preserving and Byzantine-Robust Federated Learning](https://ieeexplore.ieee.org/abstract/document/10093038) *TDSC 2023*

## Code

### Dataset processing and partitioning
The `dataset` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:
- `generate_MNIST.py`: Utility script for processing and partitioning MNIST dataset.
- `generate_FashionMNIST.py`: Utility script for processing and partitioning FashionMNIST dataset.
- `generate_Cifar10.py`: Utility script for processing and partitioning Cifar10 dataset.
- `generate_FEMNIST.py`: Utility script for processing and partitioning FEMNIST dataset.
- `generate_Cifar100.py`: Utility script for processing and partitioning Cifar100 dataset.
- `generate_Caltech256.py`: Utility script for processing and partitioning Caltech256 dataset.

An example is as follows:
```bash
python generate_MNIST.py noniid - dir
```

For specific parameter settings, please refer to [PFLlib](https://github.com/TsingZ0/PFLlib).

### Generates the key pairs

The `KGC.py` script is used to generate and store public and private key pairs for linearly homomorphic encryption and asymmetric encryption. An example is as follows:
```bash
python KGC.py 1024 20
```
- `sys.argv[1]`: The key size.
- `sys.argv[2]`: The number of clients.

### Running the Main for training

To run the `main.py` script with the default settings, you can use the following command as an example:
```bash
python -u main.py -data MNIST -m cnn -algo TrimMean -gr 200 -nc 20 -nb 10 -lbs 50 -attc 4 -attr 0.2 -go LIE_02
python -u main.py -data MNIST -m cnn -algo PEFLTheory -gr 200 -nc 20 -nb 10 -lbs 50 -attc 4 -attr 0.2 -go LIE_02
python -u main.py -data MNIST -m cnn -algo ShieldFLTheory -gr 200 -nc 20 -nb 10 -lbs 50 -attc 4 -attr 0.2 -go LIE_02
python -u main.py -data MNIST -m cnn -algo ProFLTheory -gr 200 -nc 20 -nb 10 -lbs 50 -attc 4 -attr 0.2 -go LIE_02
```
- `-algo`: The algorithm related to robust FL.
  - **Type**: str.
    - FedAvg
    - Krum
    - Median
    - TrimMean
    - Bulyan
    - FLTrust
    - VarianceReductionFL
    - PEFLTheory
    - SEARTheory
    - P2BroFLTheory
    - ShieldFLTheory
    - ProFLTheory
    - ProFLCipher
- `-attr`: The poisoning attack rate.
  - **Range**: 0 to 1.
  - **Type**: float.
- `-attc`: The poisoning attack category.
  - **Range**: 0 to 8.
  - **Type**: int.
  - **Values**:
    - 0: No attack
    - 1: Label-flipping attack
    - 2: Sign-flipping attack
    - 3: Gaussian-noise attack
    - 4: LIE attack
    - 5: Fang's attack
    - 6: Min-Max attack
    - 7: AGR-tailored attack
    - 8: FMPA

For other parameter settings, please refer to [PFLlib](https://github.com/TsingZ0/PFLlib).

## References and Acknowledgments
During the development process of this project, we referred to the following excellent projects, which provided us with valuable ideas and references for our development.

related project:

[PFLlib](https://github.com/TsingZ0/PFLlib)

[blades](https://github.com/lishenghui/blades)

[A General Framework to Evaluate Robustness of Aggregation Algorithms in Federated Learning](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)

[MODEL](https://github.com/K3ats/MODEL)

[python-paillier](https://github.com/data61/python-paillier)

[Batchcrypt](https://github.com/marcoszh/BatchCrypt)

