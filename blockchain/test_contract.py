import json
from pathlib import Path

from web3 import Web3


BASE_DIR = Path(__file__).resolve().parent.parent

GANACHE_URL = "http://127.0.0.1:8545"

DEPLOYMENT_FILE = BASE_DIR / "blockchain" / "deployment.json"
ABI_FILE = BASE_DIR / "blockchain" / "ModelRegistry_abi.json"


web3 = Web3(
    Web3.HTTPProvider(GANACHE_URL)
)

print("Connected:", web3.is_connected())
print("Chain ID:", web3.eth.chain_id)


with open(DEPLOYMENT_FILE, "r") as file:
    deployment = json.load(file)


with open(ABI_FILE, "r") as file:
    abi = json.load(file)


contract_address = deployment["contract_address"]


contract = web3.eth.contract(
    address=Web3.to_checksum_address(
        contract_address
    ),
    abi=abi
)


record_count = contract.functions.getRecordCount().call()

print("\nTotal blockchain records:", record_count)


print("\nBlockchain Records")
print("=" * 60)


for i in range(record_count):

    record = contract.functions.getRecord(i).call()

    round_number = record[0]
    stored_hash = record[1].hex()
    submitter = record[2]
    timestamp = record[3]

    print(f"\nRecord {i}")
    print("Round     :", round_number)
    print("Model Hash:", stored_hash)
    print("Submitter :", submitter)
    print("Timestamp :", timestamp)