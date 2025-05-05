import numpy as np
import phe as paillier
from phe.util import int_to_base64, base64_to_int
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from joblib import Parallel, delayed
import cryptography
import json
import os


# To best speed up the efficiency of Paillier, please refer to https://github.com/marcoszh/BatchCrypt.


def _encryption_version():
    print(paillier.__version__)
    print(cryptography.__version__)


def generate_lhe_public_private_key_pairs(key_size=1024):
    pubkey, prikey = paillier.generate_paillier_keypair(n_length=key_size)
    return pubkey, prikey


def generate_asymmetric_public_private_key_pairs(key_size=2048):
    """
    Generate a public-private key pair for asymmetric encryption (using the RSA algorithm)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )

    public_key = private_key.public_key()

    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_key_bytes, public_key_bytes


def save_asymmetric_private_key_safely(private_key_bytes, directory='./ASYMMETRIC_KEYS/PRIVATE/',
                                       filename='private_key.pem'):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as private_key_file:
        private_key_file.write(private_key_bytes)

    os.chmod(file_path, 0o600)

    print(f"The asymmetric private key has been securely saved to a file: {file_path}")


def save_asymmetric_public_key_safely(public_key_bytes, directory='./ASYMMETRIC_KEYS/PUBLIC/',
                                      filename='public_key.pem'):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as public_key_file:
        public_key_file.write(public_key_bytes)

    os.chmod(file_path, 0o600)

    print(f"The asymmetric public key has been securely saved to the file: {file_path}")


def save_lhe_private_key_safely(private_key, directory='./LHE_KEYS/', filename='lhe_key_pairs.json'):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    private_key_json = json.dumps({
        'p': int_to_base64(private_key.p),
        'q': int_to_base64(private_key.q),
        'public_key': {'n': int_to_base64(private_key.public_key.n)}
    })

    with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as f:
        f.write(private_key_json)

    print(f"The LHE key-pairs has been securely saved to the file: {file_path}")


def load_private_key_safely(filename):
    with open(filename, "r") as f:
        private_key_json = f.read()

    data = json.loads(private_key_json)
    public_key = paillier.PaillierPublicKey(n=base64_to_int(data['public_key']['n']))
    return paillier.PaillierPrivateKey(public_key, p=base64_to_int(data['p']), q=base64_to_int(data['q']))


def load_public_key_safely(filename):
    with open(filename, "r") as f:
        private_key_json = f.read()
    data = json.loads(private_key_json)
    return paillier.PaillierPublicKey(n=base64_to_int(data['public_key']['n']))


#################################################################################################
# speed up Paillier by joblib
#
# def func_encBenchmark(pubkey, nums1: List[Any]) -> List[Any]:
#     nums1_enc = Parallel(n_jobs=-1)(delayed(pubkey.encrypt)(n) for n in nums1)
#     return nums1_enc
#
#
# def func_decBenchmark(prikey, nums1_enc: List[Any]) -> List[Any]:
#     nums1 = Parallel(n_jobs=-1)(delayed(prikey.decrypt)(n) for n in nums1_enc)
#     return nums1
#
#
# def func_addBenchmark(nums1_enc: List[Any], nums2_enc: List[Any]) -> List[Any]:
#     nums_add12 = Parallel(n_jobs=-1)(delayed(lambda n1, n2: n1 + n2)(n1, n2) for n1, n2 in zip(nums1_enc, nums2_enc))
#     return nums_add12
#
#
# def func_mulBenchmark(nums1_enc: List[Any], plain: List[Any]) -> List[Any]:
#     nums_mul12 = Parallel(n_jobs=-1)(delayed(lambda n1, p: p * n1)(n1, p) for n1, p in zip(nums1_enc, plain))
#     return nums_mul12
#
#
# def func_cumsumBenchmark(nums1_enc: List[Any]):
#     n = len(nums1_enc)
#     chunk_size = 1
#     chunks = [(i, min(i + chunk_size - 1, n - 1)) for i in range(0, n, chunk_size)]
#     partial_sums_enc = Parallel(n_jobs=-1)(
#         delayed(lambda start, end, nums: sum(nums[start:end + 1]))(start, end, nums1_enc)
#         for start, end in chunks
#     )
#     cumsum_nums1_enc = sum(partial_sums_enc)
#     return cumsum_nums1_enc


#################################################################################################
# speed up Paillier by BatchCrypt
#
# def func_encBatchCrypt(pubkey, nums1: List[Any]) -> List[Any]:
#     # ...
#     return nums1_enc
#
#
# def func_decBatchCrypt(prikey, nums1_enc: List[Any]) -> List[Any]:
#     # ...
#     return nums1
#
#
# def func_addBatchCrypt(nums1_enc: List[Any], nums2_enc: List[Any]) -> List[Any]:
#     # ...
#     return nums_add12
#
#
# def func_mulBatchCrypt(nums1_enc: List[Any], plain: List[Any]) -> List[Any]:
#     # ...
#     return nums_mul12
#
#
# def func_cumsumBatchCrypt(nums1_enc: List[Any]):
#     # ...
#     return cumsum_nums1_enc

