import sqlite3
import json
from web3 import Web3


DB_PATH = "db/healthfl.db"
DEPLOYMENT_PATH = "blockchain/deployment.json"
RPC_URL = "http://127.0.0.1:8545"


# -----------------------------
# Load blockchain configuration
# -----------------------------
with open(DEPLOYMENT_PATH, "r") as f:
    deployment = json.load(f)

contract_address = deployment["contract_address"]

with open("blockchain/ModelRegistry_abi.json", "r") as f:
    abi = json.load(f)

w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise RuntimeError("Could not connect to Ganache.")

contract = w3.eth.contract(
    address=contract_address,
    abi=abi
)


# -----------------------------
# Read SQLite records
# -----------------------------
conn = sqlite3.connect(DB_PATH)

sqlite_rows = conn.execute(
    """
    SELECT round_id, global_model_hash
    FROM fl_rounds
    ORDER BY round_id
    """
).fetchall()

conn.close()


# -----------------------------
# Read blockchain records
# -----------------------------
blockchain_records = contract.functions.getAllRecords().call()


# -----------------------------
# Compare
# -----------------------------
print("=" * 70)
print("HEALTHFL-CHAIN AUDIT VERIFICATION")
print("=" * 70)

print(f"SQLite rounds      : {len(sqlite_rows)}")
print(f"Blockchain records : {len(blockchain_records)}")
print()

if len(sqlite_rows) != len(blockchain_records):
    print("❌ RECORD COUNT MISMATCH")
    raise SystemExit(1)


all_match = True

for (sqlite_round, sqlite_hash), blockchain_record in zip(
    sqlite_rows,
    blockchain_records
):

    blockchain_round = blockchain_record[0]
    blockchain_hash = blockchain_record[1].hex()

    match = (
        sqlite_round == blockchain_round
        and sqlite_hash.lower() == blockchain_hash.lower()
    )

    status = "MATCH" if match else "MISMATCH"

    print(
        f"Round {sqlite_round:2d} : {status}"
    )

    if not match:
        print(f"  SQLite     : {sqlite_hash}")
        print(f"  Blockchain : {blockchain_hash}")
        all_match = False


print()
print("=" * 70)

if all_match:
    print("✅ ALL 10 MODEL HASHES MATCH")
    print("✅ SQLITE AND BLOCKCHAIN AUDIT TRAILS ARE CONSISTENT")
else:
    print("❌ AUDIT VERIFICATION FAILED")

print("=" * 70)