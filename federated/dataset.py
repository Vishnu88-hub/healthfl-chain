import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = "data/healthcare-dataset-stroke-data.csv"

RANDOM_STATE = 42


def preprocess_patient_input(patient_data):
    """Transform raw patient fields using the experiment preprocessing."""

    df = pd.read_csv(DATA_PATH)

    numeric_columns = [
        "age",
        "avg_glucose_level",
        "bmi"
    ]

    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(df[numeric_columns])

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(df[categorical_columns])

    scaler = StandardScaler()
    imputed_numeric = pd.DataFrame(
        imputer.transform(df[numeric_columns]),
        columns=numeric_columns
    )
    scaler.fit(imputed_numeric)

    patient_frame = pd.DataFrame([patient_data])
    patient_frame[numeric_columns] = imputer.transform(
        patient_frame[numeric_columns]
    )

    encoded_categorical = encoder.transform(
        patient_frame[categorical_columns]
    )
    encoded_df = pd.DataFrame(
        encoded_categorical,
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    scaled_numeric_df = pd.DataFrame(
        scaler.transform(patient_frame[numeric_columns]),
        columns=numeric_columns
    )
    binary_df = patient_frame[
        ["hypertension", "heart_disease"]
    ].reset_index(drop=True)

    processed = pd.concat(
        [
            scaled_numeric_df,
            binary_df,
            encoded_df
        ],
        axis=1
    )

    return processed.astype(np.float32).to_numpy(), processed.columns.tolist()


def load_hospital_partitions(num_hospitals=4):

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # ---------------------------------------------------------
    # 2. Drop ID
    # ---------------------------------------------------------

    df = df.drop(columns=["id"])

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # ---------------------------------------------------------
    # 4. Define columns
    # ---------------------------------------------------------

    numeric_columns = [
        "age",
        "avg_glucose_level",
        "bmi"
    ]

    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    # ---------------------------------------------------------
    # 5. Impute missing BMI
    # ---------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X[numeric_columns] = imputer.fit_transform(
        X[numeric_columns]
    )

    # ---------------------------------------------------------
    # 6. One-hot encode categorical columns
    # ---------------------------------------------------------

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded_categorical = encoder.fit_transform(
        X[categorical_columns]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )

    encoded_df = pd.DataFrame(
        encoded_categorical,
        columns=encoded_columns,
        index=X.index
    )

    # ---------------------------------------------------------
    # 7. Scale only true numeric columns
    # ---------------------------------------------------------

    scaler = StandardScaler()

    scaled_numeric = scaler.fit_transform(
        X[numeric_columns]
    )

    scaled_numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=numeric_columns,
        index=X.index
    )

    # ---------------------------------------------------------
    # 8. Keep binary numerical columns
    # ---------------------------------------------------------

    binary_columns = [
        "hypertension",
        "heart_disease"
    ]

    binary_df = X[binary_columns].reset_index(drop=True)

    scaled_numeric_df = scaled_numeric_df.reset_index(
        drop=True
    )

    encoded_df = encoded_df.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # 9. Combine all processed features
    # ---------------------------------------------------------

    X_processed = pd.concat(
        [
            scaled_numeric_df,
            binary_df,
            encoded_df
        ],
        axis=1
    )

    X_processed = X_processed.astype(np.float32)
    feature_names = X_processed.columns.tolist()

    y = y.to_numpy(dtype=np.int64)

    # ---------------------------------------------------------
    # 10. Compute class weights from full dataset
    # ---------------------------------------------------------

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weights = weights.astype(np.float32)

    print("\nClass weights:")
    print("No Stroke:", class_weights[0])
    print("Stroke:", class_weights[1])

    # ---------------------------------------------------------
    # 11. Create non-IID hospital partitions
    # ---------------------------------------------------------

    # Create age bands
    age_bands = pd.cut(
        df["age"],
        bins=[-np.inf, 45, np.inf],
        labels=["young", "old"]
    )

    residence = df["Residence_type"]

    # Four groups:
    # 0 = Young Urban
    # 1 = Old Urban
    # 2 = Young Rural
    # 3 = Old Rural

    hospital_groups = {
        0: (age_bands == "young") & (residence == "Urban"),
        1: (age_bands == "old") & (residence == "Urban"),
        2: (age_bands == "young") & (residence == "Rural"),
        3: (age_bands == "old") & (residence == "Rural")
    }

    partitions = []

    for hospital_id in range(num_hospitals):

        mask = hospital_groups[hospital_id]

        indices = np.where(mask)[0]

        X_hospital = X_processed.iloc[
            indices
        ].to_numpy()

        y_hospital = y[indices]

        # -----------------------------------------------------
        # 12. 80/20 train-test split
        # -----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X_hospital,
            y_hospital,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_hospital
        )

        # -----------------------------------------------------
        # 13. Store hospital partition
        # -----------------------------------------------------

        hospital_data = {
            "hospital_id": hospital_id,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "class_weights": class_weights,
            "feature_names": feature_names
        }

        partitions.append(hospital_data)

        # -----------------------------------------------------
        # 14. Print hospital information
        # -----------------------------------------------------

        print(
            f"\nHospital {hospital_id}"
        )

        print(
            "Total samples:",
            len(indices)
        )

        print(
            "Training samples:",
            len(X_train)
        )

        print(
            "Testing samples:",
            len(X_test)
        )

        print(
            "Mean age:",
            df.iloc[indices]["age"].mean()
        )

        print(
            "Residence:",
            df.iloc[indices]["Residence_type"].value_counts().to_dict()
        )

        print(
            "Stroke distribution:",
            np.bincount(y_hospital)
        )

    return partitions


# -------------------------------------------------------------
# Test the dataset module directly
# -------------------------------------------------------------

if __name__ == "__main__":

    partitions = load_hospital_partitions(
        num_hospitals=4
    )

    print("\n--------------------------------")
    print("Hospital partitioning completed")
    print("--------------------------------")

    for hospital in partitions:

        print(
            f"\nHospital {hospital['hospital_id']}"
        )

        print(
            "X_train shape:",
            hospital["X_train"].shape
        )

        print(
            "y_train shape:",
            hospital["y_train"].shape
        )

        print(
            "X_test shape:",
            hospital["X_test"].shape
        )

        print(
            "y_test shape:",
            hospital["y_test"].shape
        )

        print("\nFeature names:")
        for i, feature in enumerate(partitions[0]["feature_names"]):
            print(i, feature)