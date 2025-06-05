# Byzantine-robust Federated Learning Considering Distributed Non-IID Data in Privacy-sensitive Scenarios

For federated learning (FL), its decentralized training approach is vulnerable to the targeted or untargeted attacks from Byzantine clients, which results in decreased model accuracy. In privacy-sensitive scenarios, the Byzantine clients identification is extremely difficult, due to the fact that gradient information is encrypted. To preserve privacy while identifying Byzantine clients in these scenarios, many schemes on identifying malicious local gradients over ciphertexts are proposed. However, existing works suffer from the problems of inefficiency in non-independent identically distributed (Non-IID) data and potential privacy leakage. To address the issues, we propose a privacy-preserving and byzantine-robust FL scheme (ProFL), which can resist encrypted malicious gradients effectively and protect the privacy of gradients. In ProFL, the linearly homomorphic encryption (LHE) technique is adopted to safeguard the gradient parameters. Specifically, ProFL designs two secure protocols that effectively mitigate potential threats in existing methods while maintaining computation complexity within a linear range. A robust aggregation method based on Manhattan distance is proposed to effectively enhance the robustness of federated aggregation in Non-IID data scenarios. The security and convergence proof of ProFL is given. Finally, extensive evaluations demonstrate that ProFL outperforms other privacy-preserving and byzantine-robust FL schemes in defending against various poisoning attacks.

***

This project is based on the open source project [PFLlib](https://github.com/TsingZ0/PFLlib) development.


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

### If this project helped you, please cite:

```
@article{wei2025byzantine,
  title={Byzantine-robust federated learning considering distributed Non-IID data in privacy-sensitive scenarios},
  author={Wei, Zongpu and Wang, Jinsong and Zhao, Zening and Zhao, Zhao},
  journal={Information Fusion},
  pages={103345},
  year={2025},
  publisher={Elsevier}
}
```

***

## References and Acknowledgments
During the development process of this project, we referred to the following excellent projects, which provided us with valuable ideas and references for our development.

related project:

[PFLlib](https://github.com/TsingZ0/PFLlib)

[blades](https://github.com/lishenghui/blades)

[A General Framework to Evaluate Robustness of Aggregation Algorithms in Federated Learning](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)

[MODEL](https://github.com/K3ats/MODEL)

[python-paillier](https://github.com/data61/python-paillier)

[Batchcrypt](https://github.com/marcoszh/BatchCrypt)
