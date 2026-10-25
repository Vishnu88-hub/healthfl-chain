import json
from pathlib import Path

from web3 import Web3


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Blockchain files
DEPLOYMENT_FILE = BASE_DIR / "blockchain" / "deployment.json"
ABI_FILE = BASE_DIR / "blockchain" / "ModelRegistry_abi.json"

# Ganache RPC URL
GANACHE_URL = "http://127.0.0.1:8545"


def connect_to_blockchain():
    """Connect to the local Ganache blockchain."""

    web3 = Web3(Web3.HTTPProvider(GANACHE_URL))

    if not web3.is_connected():
        raise ConnectionError(
            "Could not connect to Ganache. "
            "Make sure Ganache is running on port 8545."
        )

    return web3


def load_contract(web3):
    """Load the deployed ModelRegistry contract."""

    with open(DEPLOYMENT_FILE, "r") as file:
        deployment = json.load(file)

    with open(ABI_FILE, "r") as file:
        abi = json.load(file)

    contract_address = deployment["contract_address"]

    contract = web3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=abi
    )

    return contract


def log_round_to_chain(round_number, model_hash):
    """
    Store a model hash on the blockchain.

    Parameters
    ----------
    round_number : int
        Federated learning round number.

    model_hash : str
        64-character SHA-256 hash.
    """

    if len(model_hash) != 64:
        raise ValueError(
            "Model hash must contain exactly 64 hexadecimal characters."
        )

    try:
        bytes32_hash = bytes.fromhex(model_hash)
    except ValueError:
        raise ValueError(
            "Model hash must contain only hexadecimal characters."
        )

    if len(bytes32_hash) != 32:
        raise ValueError(
            "SHA-256 hash must convert to exactly 32 bytes."
        )

    web3 = connect_to_blockchain()

    contract = load_contract(web3)

    account = web3.eth.accounts[0]

    transaction = contract.functions.storeModelHash(
        round_number,
        bytes32_hash
    ).transact({
        "from": account
    })

    receipt = web3.eth.wait_for_transaction_receipt(transaction)

    print(
        f"Round {round_number} hash stored on blockchain."
    )
    print(
        f"Transaction hash: {receipt.transactionHash.hex()}"
    )
    print(
        f"Block number: {receipt.blockNumber}"
    )

    return {
        "transaction_hash": receipt.transactionHash.hex(),
        "block_number": receipt.blockNumber
    }


if __name__ == "__main__":

    # Test hash generated earlier
    test_hash = (
        "353a1ea64fb58a7d648e9852d4122f6c"
        "169816fecf0e72fbf396832762dd09d5"
    )

    print("=" * 60)
    print("BLOCKCHAIN LOGGER TEST")
    print("=" * 60)

    tx_hash = log_round_to_chain(
        round_number=99,
        model_hash=test_hash
    )

    print("\nSUCCESS")
    print("Transaction:", tx_hash)