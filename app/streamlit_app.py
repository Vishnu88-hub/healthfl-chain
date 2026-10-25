import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from blockchain.logger import connect_to_blockchain, load_contract
from db.db_utils import (
    DB_PATH,
    get_connection,
    get_predictions,
    initialize_database,
    insert_prediction,
)
from federated.dataset import (
    load_hospital_partitions,
    preprocess_patient_input,
)
from federated.explainability import (
    create_shap_explainer,
    explain_single_prediction,
    load_global_model,
    predict_probability,
)


MODEL_PATH = BASE_DIR / "saved_models" / "final_global_model.pth"
GLOBAL_SHAP_PATH = BASE_DIR / "saved_models" / "shap_global_importance.png"
INDIVIDUAL_SHAP_PATH = (
    BASE_DIR / "saved_models" / "shap_individual_prediction.png"
)
NUM_HOSPITALS = 4
DEFAULT_ROUNDS = 10
DEFAULT_LOCAL_EPOCHS = 1
CENTRALIZED_BASELINE = {
    "accuracy": 0.7803,
    "precision": 0.1429,
    "recall": 0.7000,
    "f1": 0.2373,
    "auc": 0.7918,
}


def query_rows(query, parameters=()):
    if not DB_PATH.exists():
        initialize_database()
    connection = get_connection()
    try:
        return connection.execute(query, parameters).fetchall()
    finally:
        connection.close()


@st.cache_resource
def get_model_context():
    return load_global_model()


@st.cache_data
def get_partitions():
    return load_hospital_partitions(num_hospitals=NUM_HOSPITALS)


@st.cache_resource
def get_shap_context():
    model, feature_names, round_number = get_model_context()
    partitions = get_partitions()
    training_data = np.concatenate(
        [hospital["X_train"] for hospital in partitions],
        axis=0,
    )
    rng = np.random.default_rng(42)
    size = min(50, len(training_data))
    background = training_data[
        rng.choice(len(training_data), size=size, replace=False)
    ]
    return (
        model,
        create_shap_explainer(model, background),
        feature_names,
        round_number,
    )


def database_available():
    try:
        initialize_database()
        return True
    except Exception:
        return False


def get_rounds():
    return query_rows(
        """
        SELECT round_id, started_at, completed_at, num_hospitals,
               global_model_hash, blockchain_tx_hash,
               blockchain_block_number, status
        FROM fl_rounds
        ORDER BY round_id DESC
        """
    )


def get_latest_hospital_metrics(round_id):
    return query_rows(
        """
        SELECT hospital_id, num_test_samples, accuracy, precision,
               recall, f1, auc
        FROM hospital_round_metrics
        WHERE round_id = ?
        ORDER BY hospital_id
        """,
        (round_id,),
    )


def aggregate_metrics(rows):
    if not rows:
        return {}
    total = sum(row[1] or 0 for row in rows)
    if not total:
        return {}
    names = ("accuracy", "precision", "recall", "f1", "auc")
    return {
        name: sum((row[index] or 0) * (row[1] or 0) for row in rows) / total
        for index, name in zip(range(2, 7), names)
    }


def blockchain_records():
    web3 = connect_to_blockchain()
    contract = load_contract(web3)
    records = contract.functions.getAllRecords().call()
    normalized = []
    for record in records:
        if hasattr(record, "round"):
            round_number = record.round
            model_hash = record.modelHash
            submitter = record.submitter
            timestamp = record.timestamp
        else:
            round_number, model_hash, submitter, timestamp = record
        if isinstance(model_hash, bytes):
            model_hash = model_hash.hex()
        normalized.append(
            {
                "round": int(round_number),
                "model_hash": str(model_hash).removeprefix("0x"),
                "submitter": submitter,
                "timestamp": int(timestamp),
            }
        )
    return web3, normalized


def check_blockchain():
    web3 = connect_to_blockchain()
    contract = load_contract(web3)
    contract.functions.getRecordCount().call()
    return web3


def status_badge(label, available, detail=""):
    icon = "✓" if available else "!"
    state = "Ready" if available else "Not available"
    st.markdown(f"**{icon} {label}:** {state} {detail}")


def render_overview():
    st.title("HealthFL-Chain")
    st.subheader(
        "Blockchain-Enabled Federated Learning for "
        "Privacy-Preserving Disease Prediction"
    )
    st.info(
        "This application is for research and educational purposes only. "
        "It is not a medical diagnostic tool and should not be used for "
        "clinical decision-making."
    )

    st.header("System status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_badge("Model", MODEL_PATH.exists())
    with col2:
        status_badge("SQLite", database_available())
    with col3:
        status_badge("SHAP", GLOBAL_SHAP_PATH.exists())
    with col4:
        try:
            check_blockchain()
            status_badge("Ganache", True)
        except Exception:
            status_badge("Ganache", False)
            st.caption("Start Ganache and deploy the configured contract.")

    rounds = get_rounds() if database_available() else []
    latest = rounds[0] if rounds else None
    metrics = aggregate_metrics(
        get_latest_hospital_metrics(latest[0]) if latest else []
    )
    st.header("Research results")
    metric_columns = st.columns(6)
    values = [
        ("Federated ROC-AUC", metrics.get("auc")),
        ("Federated F1", metrics.get("f1")),
        ("Centralized ROC-AUC", CENTRALIZED_BASELINE["auc"]),
        ("Centralized F1", CENTRALIZED_BASELINE["f1"]),
        ("FL rounds", latest[0] if latest else None),
        ("Hospitals", latest[3] if latest else None),
    ]
    for column, (label, value) in zip(metric_columns, values):
        column.metric(label, "Not available" if value is None else (
            f"{value:.4f}" if isinstance(value, float) else str(value)
        ))

    if latest:
        st.caption(
            f"Latest recorded round: {latest[0]} | "
            f"Status: {latest[7]} | Model hash: {latest[4]}"
        )
    else:
        st.warning("No federated-learning rounds are recorded yet.")


def render_train():
    st.title("Train")
    st.warning(
        "Training is a local, single-machine simulation of four hospitals. "
        "It does not represent a distributed clinical deployment."
    )
    rounds = st.number_input(
        "FL rounds", min_value=1, max_value=100, value=DEFAULT_ROUNDS
    )
    epochs = st.number_input(
        "Local epochs", min_value=1, max_value=20, value=DEFAULT_LOCAL_EPOCHS
    )
    record_chain = st.checkbox(
        "Write aggregated hashes to Ganache",
        value=False,
        help="Leave disabled to preserve model-only training when Ganache is unavailable.",
    )

    process = st.session_state.get("training_process")
    if process is not None and process.poll() is None:
        st.info("Training is running. Refresh this page to update its status.")
        return
    if process is not None:
        code = process.returncode
        st.session_state.pop("training_process", None)
        if code == 0:
            st.success("Training completed.")
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            st.error(f"Training stopped with exit code {code}. Check the terminal output.")

    if st.button("Start training", type="primary"):
        if record_chain:
            try:
                check_blockchain()
            except Exception as exc:
                st.error(
                    "Blockchain unavailable. Start Ganache on port 8545 "
                    f"before enabling blockchain recording: {exc}"
                )
                return
        environment = os.environ.copy()
        environment["HEALTHFL_NUM_ROUNDS"] = str(rounds)
        environment["HEALTHFL_LOCAL_EPOCHS"] = str(epochs)
        environment["HEALTHFL_SAVE_MODEL_ONLY"] = "0" if record_chain else "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "federated.server"],
            cwd=BASE_DIR,
            env=environment,
        )
        st.session_state["training_process"] = process
        st.success(
            f"Started {rounds}-round training with {epochs} local epoch(s). "
            "The process is running outside the Streamlit request."
        )

    latest = get_rounds()[0] if database_available() and get_rounds() else None
    if latest:
        st.subheader("Latest recorded round")
        st.write(
            f"Round {latest[0]} — {latest[7]} — completed at {latest[2]}"
        )


def render_audit():
    st.title("Audit Trail")
    rounds = get_rounds() if database_available() else []
    if not rounds:
        st.info("No SQLite audit records are available.")
        return
    frame = pd.DataFrame(
        rounds,
        columns=[
            "Round", "Started", "Completed", "Hospitals", "Model Hash",
            "Transaction Hash", "Block Number", "Status",
        ],
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect round", frame["Round"].tolist())
    row = next(item for item in rounds if item[0] == selected)
    st.markdown(
        "**FL round** → **global model** → **SHA-256 hash** → "
        "**blockchain transaction** → **block number**"
    )
    st.json(
        {
            "round": row[0],
            "global_model_hash": row[4],
            "blockchain_transaction": row[5],
            "block_number": row[6],
            "status": row[7],
        }
    )
    if st.button("Verify round"):
        try:
            web3, records = blockchain_records()
            matches = [
                record for record in records if record["round"] == selected
            ]
            hash_match = bool(matches) and (
                matches[-1]["model_hash"].lower() == row[4].lower()
            )
            block_match = False
            if row[5]:
                receipt = web3.eth.get_transaction_receipt(row[5])
                block_match = int(receipt["blockNumber"]) == int(row[6])
            if hash_match and block_match:
                st.success("SQLite hash and transaction block were verified on Ganache.")
            else:
                st.error("Verification failed: the records do not fully match.")
        except Exception as exc:
            st.error(f"Blockchain unavailable: {exc}")


def render_insights():
    st.title("Model Insights")
    st.write(
        "Higher mean absolute SHAP value means the feature had a stronger "
        "influence on model predictions in the analyzed sample. This is not "
        "evidence of medical causation."
    )
    if GLOBAL_SHAP_PATH.exists():
        st.image(str(GLOBAL_SHAP_PATH), caption="Global SHAP importance")
    else:
        st.info("Global SHAP image is not available.")

    rounds = get_rounds() if database_available() else []
    federated = {}
    if rounds:
        federated = aggregate_metrics(get_latest_hospital_metrics(rounds[0][0]))
    comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Federated": [
                federated.get(key, np.nan)
                for key in ("accuracy", "precision", "recall", "f1", "auc")
            ],
            "Centralized": [
                CENTRALIZED_BASELINE[key]
                for key in ("accuracy", "precision", "recall", "f1", "auc")
            ],
        }
    )
    st.subheader("Federated vs centralized baseline")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index("Metric")[["Federated", "Centralized"]])

    if st.button("Calculate ranked global SHAP values"):
        with st.spinner("Calculating SHAP values from the existing model..."):
            _, explainer, feature_names, _ = get_shap_context()
            partitions = get_partitions()
            data = np.concatenate(
                [hospital["X_train"] for hospital in partitions], axis=0
            )[:20]
            values = explainer.shap_values(data, nsamples=100)
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, 0]
            importance = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean absolute SHAP": np.mean(np.abs(values), axis=0),
                }
            ).sort_values("Mean absolute SHAP", ascending=False)
            st.dataframe(importance, use_container_width=True, hide_index=True)


def render_predict():
    st.title("Predict")
    st.info(
        "This prediction is a research demonstration and must not be used "
        "for clinical decision-making."
    )
    with st.form("patient_form"):
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart disease", [0, 1])
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
        work_type = st.selectbox(
            "Work type",
            ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
        )
        residence = st.selectbox("Residence type", ["Rural", "Urban"])
        glucose = st.number_input(
            "Average glucose level", min_value=0.0, value=100.0
        )
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        smoking = st.selectbox(
            "Smoking status",
            ["Unknown", "formerly smoked", "never smoked", "smokes"],
        )
        submitted = st.form_submit_button("Run prediction", type="primary")

    if not submitted:
        return
    patient = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }
    try:
        processed, feature_names = preprocess_patient_input(patient)
        model, explainer, _, round_number = get_shap_context()
        prediction, probability, explanation = explain_single_prediction(
            model, explainer, processed[0], feature_names
        )
        st.session_state["last_prediction"] = (
            prediction, probability, explanation, round_number
        )
        insert_prediction(
            prediction=prediction,
            probability=float(probability),
            input_data=json.dumps(patient),
            explanation=json.dumps(
                explanation.head(10).to_dict(orient="records")
            ),
        )
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        return

    prediction, probability, explanation, round_number = (
        st.session_state["last_prediction"]
    )
    st.metric("Predicted class", "Stroke risk" if prediction else "No stroke risk")
    st.metric("Stroke probability", f"{probability:.4f}")
    st.caption(f"Round-{round_number} global model")
    st.subheader("Top factors influencing this prediction")
    st.dataframe(
        explanation.head(10)[["feature", "value", "shap_value"]],
        use_container_width=True,
        hide_index=True,
    )
    if INDIVIDUAL_SHAP_PATH.exists():
        st.image(str(INDIVIDUAL_SHAP_PATH), caption="Individual SHAP explanation")


def main():
    st.set_page_config(
        page_title="HealthFL-Chain",
        page_icon="🧬",
        layout="wide",
    )
    st.sidebar.title("HealthFL-Chain")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Train", "Audit Trail", "Model Insights", "Predict"],
    )
    st.sidebar.caption("Local academic/research prototype")
    if page == "Overview":
        render_overview()
    elif page == "Train":
        render_train()
    elif page == "Audit Trail":
        render_audit()
    elif page == "Model Insights":
        render_insights()
    else:
        render_predict()


if __name__ == "__main__":
    main()