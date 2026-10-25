-- ============================================
-- HealthFL-Chain SQLite Database Schema
-- ============================================

PRAGMA foreign_keys = ON;


-- ============================================
-- FL Round Information
-- ============================================

CREATE TABLE IF NOT EXISTS fl_rounds (

    round_id INTEGER PRIMARY KEY,

    started_at TEXT,

    completed_at TEXT,

    num_hospitals INTEGER NOT NULL,

    global_model_hash TEXT,

    blockchain_tx_hash TEXT,

    blockchain_block_number INTEGER,

    status TEXT NOT NULL

);


-- ============================================
-- Hospital-Level Metrics
-- ============================================

CREATE TABLE IF NOT EXISTS hospital_round_metrics (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    round_id INTEGER NOT NULL,

    hospital_id INTEGER NOT NULL,

    num_train_samples INTEGER,

    num_test_samples INTEGER,

    loss REAL,

    accuracy REAL,

    precision REAL,

    recall REAL,

    f1 REAL,

    auc REAL,

    FOREIGN KEY (round_id)
        REFERENCES fl_rounds(round_id)
        ON DELETE CASCADE

);


-- ============================================
-- Prediction Logs
-- ============================================

CREATE TABLE IF NOT EXISTS predictions_log (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    created_at TEXT,

    prediction INTEGER,

    probability REAL,

    input_data TEXT,

    explanation TEXT

);


-- ============================================
-- Useful Indexes
-- ============================================

CREATE INDEX IF NOT EXISTS idx_hospital_metrics_round

ON hospital_round_metrics(round_id);


CREATE INDEX IF NOT EXISTS idx_predictions_created

ON predictions_log(created_at);