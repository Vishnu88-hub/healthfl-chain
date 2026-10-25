from pathlib import Path

from web3 import Web3
from solcx import compile_source, install_solc


# ============================================================
# CONFIGURATION
# ============================================================

GANACHE_URL = "http://127.0.0.1:8545"
SOLIDITY_VERSION = "0.8.19"


# ============================================================
# CONNECT TO GANACHE
# ============================================================

w3 = Web3(
    Web3.HTTPProvider(GANACHE_URL)
)

if not w3.is_connected():
    raise ConnectionError(
        "Could not connect to Ganache."
    )

print("Connected to Ganache")
print("Chain ID:", w3.eth.chain_id)


# ============================================================
# GET GANACHE ACCOUNT
# ============================================================

account = w3.eth.accounts[0]

print("Deploying from account:")
print(account)


# ============================================================
# READ SMART CONTRACT
# ============================================================

contract_path = (
    Path(__file__).resolve().parent.parent
    / "contracts"
    / "ModelRegistry.sol"
)

with open(
    contract_path,
    "r",
    encoding="utf-8"
) as file:
    contract_source = file.read()


# ============================================================
# INSTALL SOLIDITY COMPILER
# ============================================================

print("\nInstalling Solidity compiler...")

install_solc(SOLIDITY_VERSION)


# ============================================================
# COMPILE SMART CONTRACT
# ============================================================

print("Compiling ModelRegistry.sol...")

compiled = compile_source(
    contract_source,
    solc_version=SOLIDITY_VERSION,
    output_values=[
        "abi",
        "bin"
    ]
)

contract_id = (
    "<stdin>:ModelRegistry"
)

contract_interface = compiled[
    contract_id
]

abi = contract_interface["abi"]
bytecode = contract_interface["bin"]


# ============================================================
# CREATE CONTRACT OBJECT
# ============================================================

ModelRegistry = w3.eth.contract(
    abi=abi,
    bytecode=bytecode
)


# ============================================================
# DEPLOY CONTRACT
# ============================================================

print("\nDeploying ModelRegistry...")

transaction_hash = ModelRegistry.constructor().transact({
    "from": account
})


# ============================================================
# WAIT FOR TRANSACTION
# ============================================================

print("Waiting for deployment...")

transaction_receipt = (
    w3.eth.wait_for_transaction_receipt(
        transaction_hash
    )
)


# ============================================================
# GET CONTRACT ADDRESS
# ============================================================

contract_address = (
    transaction_receipt.contractAddress
)


print("\n" + "=" * 60)
print("SMART CONTRACT DEPLOYED SUCCESSFULLY")
print("=" * 60)

print("Contract address:")
print(contract_address)

print("\nTransaction hash:")
print(transaction_hash.hex())

print("\nBlock number:")
print(transaction_receipt.blockNumber)


# ============================================================
# SAVE ABI
# ============================================================

import json

abi_path = (
    Path(__file__).resolve().parent
    / "ModelRegistry_abi.json"
)

with open(
    abi_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        abi,
        file,
        indent=4
    )


# ============================================================
# SAVE DEPLOYMENT INFORMATION
# ============================================================

deployment_path = (
    Path(__file__).resolve().parent
    / "deployment.json"
)

deployment_data = {
    "contract_address": contract_address,
    "chain_id": w3.eth.chain_id,
    "deployer": account,
    "abi_file": "ModelRegistry_abi.json"
}

with open(
    deployment_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        deployment_data,
        file,
        indent=4
    )


print("\nDeployment information saved.")

print(
    f"\nABI: {abi_path}"
)

print(
    f"Deployment: {deployment_path}"
)