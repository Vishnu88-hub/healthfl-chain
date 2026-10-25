import numpy as np
import pandas as pd
import torch
import shap

from federated.model import StrokeMLP
from federated.dataset import load_hospital_partitions


MODEL_PATH = "saved_models/final_global_model.pth"

BACKGROUND_SIZE = 50
EXPLANATION_SIZE = 20
SHAP_NSAMPLES = 100


# ---------------------------------------------------------
# Load trained global model
# ---------------------------------------------------------

def load_global_model():

    checkpoint = torch.load(
        MODEL_PATH,
        weights_only=False
    )

    feature_names = checkpoint["feature_names"]

    model = StrokeMLP(
        input_size=len(feature_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, feature_names, checkpoint["round"]


# ---------------------------------------------------------
# Model prediction function for SHAP
# ---------------------------------------------------------

def predict_probability(model, X):

    X = np.asarray(X, dtype=np.float32)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        outputs = model(X_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

    return probabilities.numpy()


# ---------------------------------------------------------
# Create SHAP explainer
# ---------------------------------------------------------

def create_shap_explainer(
    model,
    background_data
):

    def model_predict(X):

        return predict_probability(
            model,
            X
        )

    explainer = shap.KernelExplainer(
        model_predict,
        background_data
    )

    return explainer


# ---------------------------------------------------------
# Global SHAP feature importance
# ---------------------------------------------------------

def calculate_global_importance():

    print("\n" + "=" * 60)
    print("SHAP GLOBAL FEATURE IMPORTANCE")
    print("=" * 60)

    # Load trained global model
    model, feature_names, round_number = (
        load_global_model()
    )

    print(
        f"\nLoaded global model from Round {round_number}"
    )

    print(
        f"Number of features: {len(feature_names)}"
    )

    # Load hospital partitions
    partitions = load_hospital_partitions(
        num_hospitals=4
    )

    # Combine training data from all hospitals
    X_train_all = np.concatenate(
        [
            hospital["X_train"]
            for hospital in partitions
        ],
        axis=0
    )

    print(
        "\nCombined training samples:",
        len(X_train_all)
    )

    # -----------------------------------------------------
    # Select background data
    # -----------------------------------------------------

    rng = np.random.default_rng(42)

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train_all)
    )

    background_indices = rng.choice(
        len(X_train_all),
        size=background_size,
        replace=False
    )

    background_data = X_train_all[
        background_indices
    ]

    print(
        "SHAP background samples:",
        len(background_data)
    )

    # -----------------------------------------------------
    # Select samples to explain
    # -----------------------------------------------------

    explanation_size = min(
        EXPLANATION_SIZE,
        len(X_train_all)
    )

    explanation_indices = rng.choice(
        len(X_train_all),
        size=explanation_size,
        replace=False
    )

    explanation_data = X_train_all[
        explanation_indices
    ]

    print(
        "Samples to explain:",
        len(explanation_data)
    )

    # -----------------------------------------------------
    # Create SHAP explainer
    # -----------------------------------------------------

    print(
        "\nCreating SHAP KernelExplainer..."
    )

    explainer = create_shap_explainer(
        model=model,
        background_data=background_data
    )

    # -----------------------------------------------------
    # Calculate SHAP values
    # -----------------------------------------------------

    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        explanation_data,
        nsamples=SHAP_NSAMPLES
    )

    shap_values = np.asarray(
        shap_values
    )

    print(
        "SHAP values shape:",
        shap_values.shape
    )

    # -----------------------------------------------------
    # Handle SHAP output shape
    # -----------------------------------------------------

    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 0]

    # -----------------------------------------------------
    # Calculate mean absolute SHAP
    # -----------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })

    importance = importance.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print(
        "\nGlobal Feature Importance:"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    return (
        importance,
        explainer,
        shap_values,
        explanation_data,
        feature_names
    )


def save_global_importance_plot(
    importance,
    output_path="saved_models/shap_global_importance.png",
    top_n=10
):
    import matplotlib.pyplot as plt

    top_features = importance.head(top_n)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"][::-1],
        top_features["mean_abs_shap"][::-1]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(
        "Global SHAP Feature Importance - "
        "Round 10 Federated Model"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSHAP global importance plot saved to: "
        f"{output_path}"
    )


def explain_single_prediction(
    model,
    explainer,
    sample,
    feature_names
):
    """
    Generate a SHAP explanation for one prediction.
    """

    sample = np.asarray(
        sample,
        dtype=np.float32
    ).reshape(1, -1)

    # Model prediction
    probability = predict_probability(
        model,
        sample
    )[0]

    prediction = int(
        probability >= 0.5
    )

    # SHAP explanation
    shap_values = explainer.shap_values(
        sample,
        nsamples=SHAP_NSAMPLES
    )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    shap_values = shap_values[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": sample[0],
        "shap_value": shap_values
    })

    explanation["abs_shap_value"] = (
        np.abs(
            explanation["shap_value"]
        )
    )

    explanation = explanation.sort_values(
        by="abs_shap_value",
        ascending=False
    ).reset_index(drop=True)

    return (
        prediction,
        probability,
        explanation
    )


def test_single_prediction():

    print("\n" + "=" * 60)
    print("SHAP INDIVIDUAL PREDICTION EXPLANATION")
    print("=" * 60)

    # Load model
    model, feature_names, round_number = (
        load_global_model()
    )

    # Load hospital partitions
    partitions = load_hospital_partitions(
        num_hospitals=4
    )

    # Use first test sample from Hospital 1
    sample = partitions[1]["X_test"][0]

    # Create background dataset
    X_train_all = np.concatenate(
        [
            hospital["X_train"]
            for hospital in partitions
        ],
        axis=0
    )

    rng = np.random.default_rng(42)

    background_indices = rng.choice(
        len(X_train_all),
        size=min(
            BACKGROUND_SIZE,
            len(X_train_all)
        ),
        replace=False
    )

    background_data = X_train_all[
        background_indices
    ]

    # Create SHAP explainer
    explainer = create_shap_explainer(
        model=model,
        background_data=background_data
    )

    # Explain prediction
    prediction, probability, explanation = (
        explain_single_prediction(
            model=model,
            explainer=explainer,
            sample=sample,
            feature_names=feature_names
        )
    )

    print(
        f"\nModel Round: {round_number}"
    )

    print(
        f"Predicted class: {prediction}"
    )

    print(
        f"Stroke probability: {probability:.4f}"
    )

    print("\nFeature contributions:")

    print(
        explanation.to_string(
            index=False
        )
    )

    save_individual_shap_plot(
        explanation=explanation,
        prediction=prediction,
        probability=probability,
        top_n=10
    )

    return (
        prediction,
        probability,
        explanation
    )


def save_individual_shap_plot(
    explanation,
    prediction,
    probability,
    output_path="saved_models/shap_individual_prediction.png",
    top_n=10
):
    import matplotlib.pyplot as plt

    top_features = (
        explanation
        .head(top_n)
        .sort_values(
            by="shap_value",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"],
        top_features["shap_value"]
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.xlabel(
        "SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Individual SHAP Explanation\n"
        f"Prediction: {prediction} | "
        f"Stroke Probability: {probability:.4f}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nIndividual SHAP plot saved to: "
        f"{output_path}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

# if __name__ == "__main__":

#     importance, _, _, _, _ = (
#         calculate_global_importance()
#     )

#     save_global_importance_plot(
#         importance=importance,
#         top_n=10
#     )

if __name__ == "__main__":

    test_single_prediction()