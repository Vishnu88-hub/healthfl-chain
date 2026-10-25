# HealthFL-Chain

HealthFL-Chain is an academic research prototype demonstrating
blockchain-enabled federated learning for privacy-preserving stroke prediction.
It simulates four hospitals on one machine using the public
`healthcare-dataset-stroke-data.csv` dataset.

## Research motivation

Hospitals may want to collaborate on machine-learning models without
exchanging their local training records. This project demonstrates a
single-machine federated-learning workflow and records aggregated model
integrity hashes in a local Ethereum test blockchain.

## Architecture

The Streamlit dashboard calls the existing federated-learning, PyTorch,
SQLite, blockchain, and SHAP modules:

```text
Patient input -> shared preprocessing -> Round-10 model -> prediction + SHAP
                                  ^
Four hospital partitions -> FedAvg -> SHA-256 -> Ganache + SQLite audit
```

## Dataset and preprocessing

The project uses the 5,110-row stroke dataset in `data/`. The existing
pipeline drops `id`, median-imputes `bmi`, one-hot encodes categorical
features, standardizes `age`, `avg_glucose_level`, and `bmi`, keeps the two
binary features unchanged, and creates four deterministic non-IID partitions.
The final input has 21 features and uses random state 42.

## Federated learning

The default experiment uses four simulated hospitals, ten Flower rounds, one
local epoch, FedAvg aggregation, and seed 42. The MLP is 21 -> 64 -> 32 -> 2.
The centralized baseline remains the reference experiment and is not retuned.

## Blockchain and audit storage

Ganache supplies a local Ethereum test chain at `http://127.0.0.1:8545`.
`contracts/ModelRegistry.sol` stores the round, model hash, submitter, and
timestamp. The server hashes aggregated model parameters with SHA-256.
SQLite stores round records, hospital metrics, and prediction logs in
`db/healthfl.db`. The dashboard only reads blockchain records and verifies
them against SQLite; it does not create arbitrary audit records.

## Explainability

The saved Round-10 model is `saved_models/final_global_model.pth`.
Existing global and individual SHAP plots are displayed by the dashboard.
SHAP influence is an explanation of the model's analyzed samples, not medical
causation.

## Dashboard

Launch `app/streamlit_app.py` to use:

1. **Overview** — system status and actual recorded research metrics.
2. **Train** — controlled local simulation with selectable rounds and epochs.
3. **Audit Trail** — SQLite records, blockchain records, and round verification.
4. **Model Insights** — SHAP results and federated versus centralized metrics.
5. **Predict** — raw patient fields, model probability, SHAP explanation, and
   prediction logging through `insert_prediction`.

## Installation

Use Python 3.10 or newer:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the project

Initialize SQLite:

```powershell
python -m db.db_utils
```

Start Ganache on port 8545, then deploy the contract if a deployment is not
already present:

```powershell
python blockchain\deploy.py
```

Run federated learning with the finalized defaults:

```powershell
python -m federated.server
```

Launch the dashboard:

```powershell
streamlit run app\streamlit_app.py
```

The dashboard remains usable when optional artifacts are missing. It reports
missing models, SHAP files, SQLite data, or Ganache instead of displaying
invented values.

## Expected results

The tested centralized baseline is accuracy 0.7803, precision 0.1429, recall
0.7000, F1 0.2373, and ROC-AUC 0.7918. Federated metrics are read from the
latest actual hospital records in SQLite and are never synthesized by the UI.

## Limitations and future work

This is a single-machine simulation using a local Ethereum test blockchain and
no real patient data. It is not a production deployment and does not provide
differential privacy, secure aggregation, real distributed hospitals, clinical
validation, or clinical decision support. Future work could evaluate secure
aggregation, privacy guarantees, external blockchain infrastructure, stronger
calibration, and multi-site deployment.
