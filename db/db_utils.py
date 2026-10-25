import sqlite3
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "db" / "healthfl.db"

SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def get_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ---------------------------------------------------------
# Initialize database
# ---------------------------------------------------------

def initialize_database():

    with open(SCHEMA_PATH, "r") as file:
        schema = file.read()

    connection = get_connection()

    try:

        connection.executescript(schema)

        connection.commit()

    finally:

        connection.close()


# ---------------------------------------------------------
# FL round operations
# ---------------------------------------------------------

def insert_fl_round(
    round_id,
    num_hospitals,
    global_model_hash,
    blockchain_tx_hash,
    blockchain_block_number,
    status
):

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO fl_rounds (
                round_id,
                started_at,
                completed_at,
                num_hospitals,
                global_model_hash,
                blockchain_tx_hash,
                blockchain_block_number,
                status
            )
            VALUES (
                ?,
                datetime('now'),
                datetime('now'),
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                round_id,
                num_hospitals,
                global_model_hash,
                blockchain_tx_hash,
                blockchain_block_number,
                status
            )
        )

        connection.commit()

    finally:

        connection.close()


def get_fl_round(round_id):

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                round_id,
                started_at,
                completed_at,
                num_hospitals,
                global_model_hash,
                blockchain_tx_hash,
                blockchain_block_number,
                status
            FROM fl_rounds
            WHERE round_id = ?
            """,
            (round_id,)
        )

        return cursor.fetchone()

    finally:

        connection.close()


# ---------------------------------------------------------
# Hospital metrics operations
# ---------------------------------------------------------

def insert_hospital_metrics(
    round_id,
    hospital_id,
    num_train_samples,
    num_test_samples,
    loss,
    accuracy,
    precision,
    recall,
    f1,
    auc
):

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO hospital_round_metrics (
                round_id,
                hospital_id,
                num_train_samples,
                num_test_samples,
                loss,
                accuracy,
                precision,
                recall,
                f1,
                auc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                hospital_id,
                num_train_samples,
                num_test_samples,
                loss,
                accuracy,
                precision,
                recall,
                f1,
                auc
            )
        )

        connection.commit()

    finally:

        connection.close()


def get_hospital_metrics(round_id):

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                round_id,
                hospital_id,
                num_train_samples,
                num_test_samples,
                loss,
                accuracy,
                precision,
                recall,
                f1,
                auc
            FROM hospital_round_metrics
            WHERE round_id = ?
            ORDER BY hospital_id
            """,
            (round_id,)
        )

        return cursor.fetchall()

    finally:

        connection.close()


# ---------------------------------------------------------
# Prediction operations
# ---------------------------------------------------------

def insert_prediction(
    prediction,
    probability,
    input_data,
    explanation
):

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO predictions_log (
                created_at,
                prediction,
                probability,
                input_data,
                explanation
            )
            VALUES (
                datetime('now'),
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                prediction,
                probability,
                input_data,
                explanation
            )
        )

        connection.commit()

    finally:

        connection.close()


def get_predictions():

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                id,
                created_at,
                prediction,
                probability,
                input_data,
                explanation
            FROM predictions_log
            ORDER BY id
            """
        )

        return cursor.fetchall()

    finally:

        connection.close()


# ---------------------------------------------------------
# Initialize database when run directly
# ---------------------------------------------------------

if __name__ == "__main__":

    initialize_database()

    print("=" * 60)
    print("HEALTHFL-CHAIN SQLITE DATABASE")
    print("=" * 60)

    print("\nDatabase initialized successfully.")

    print("\nDatabase location:")
    print(DB_PATH)

    print("\nTables:")

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )

        tables = cursor.fetchall()

        for table in tables:
            print(f" - {table[0]}")

    finally:

        connection.close()