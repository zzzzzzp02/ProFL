import sys
from utils.encryption import *

key_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# KGC: Generate and store LHE key pairs
pubkey, prikey = generate_lhe_public_private_key_pairs(key_size=key_size)
save_lhe_private_key_safely(private_key=prikey, directory='./flcore/LHE_KEYS/', filename=f'lhe_key_pairs.json')

# KGC: Generate and store Asymmetric key pairs for all clients
for n in range(num_clients):
    private_key_bytes, public_key_bytes = generate_asymmetric_public_private_key_pairs(key_size=key_size)
    save_asymmetric_private_key_safely(private_key_bytes, directory='./flcore/ASYMMETRIC_KEYS/PUBLIC/', filename=f'private_key_{n}.pem')
    save_asymmetric_public_key_safely(public_key_bytes, directory='./flcore/ASYMMETRIC_KEYS/PRIVATE/', filename=f'public_key_{n}.pem')

