import numpy as np

from dataset import load_hospital_partitions
from model import StrokeMLP
from trainer import train_model
from evaluate import evaluate_model


def main():

    print("=" * 60)
    print("CENTRALIZED BASELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load all hospital partitions
    # ---------------------------------------------------------

    partitions = load_hospital_partitions(
        num_hospitals=4
    )

    # ---------------------------------------------------------
    # 2. Combine training data from all hospitals
    # ---------------------------------------------------------

    X_train_list = []
    y_train_list = []

    for hospital in partitions:

        X_train_list.append(
            hospital["X_train"]
        )

        y_train_list.append(
            hospital["y_train"]
        )

    X_train = np.concatenate(
        X_train_list,
        axis=0
    )

    y_train = np.concatenate(
        y_train_list,
        axis=0
    )

    print("\nCombined training data")
    print("----------------------")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # ---------------------------------------------------------
    # 3. Use the class weights calculated from full dataset
    # ---------------------------------------------------------

    class_weights = partitions[0]["class_weights"]

    print("\nClass weights:")
    print("No Stroke:", class_weights[0])
    print("Stroke:", class_weights[1])

    # ---------------------------------------------------------
    # 4. Create the MLP
    # ---------------------------------------------------------

    input_size = X_train.shape[1]

    print("\nInput features:", input_size)

    model = StrokeMLP(
        input_size=input_size
    )

    print("\nModel created successfully.")

    # ---------------------------------------------------------
    # 5. Train centralized model
    # ---------------------------------------------------------

    print("\nStarting centralized training...\n")

    model, final_loss = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        class_weights=class_weights,
        epochs=30,
        learning_rate=0.001,
        batch_size = 64
    )

    print("\nTraining completed.")
    print("Final training loss:", final_loss)

    # ---------------------------------------------------------
    # 6. Evaluate separately on every hospital
    # ---------------------------------------------------------

    all_y_test = []
    all_predictions = []

    print("\n")
    print("=" * 60)
    print("HOSPITAL TEST RESULTS")
    print("=" * 60)

    for hospital in partitions:

        hospital_id = hospital["hospital_id"]

        metrics = evaluate_model(
            model=model,
            X_test=hospital["X_test"],
            y_test=hospital["y_test"]
        )

        print(f"\nHospital {hospital_id}")
        print("----------------------")

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1       : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['auc']:.4f}"
        )

    # ---------------------------------------------------------
    # 7. Evaluate on combined test data
    # ---------------------------------------------------------

    X_test_list = []
    y_test_list = []

    for hospital in partitions:

        X_test_list.append(
            hospital["X_test"]
        )

        y_test_list.append(
            hospital["y_test"]
        )

    X_test = np.concatenate(
        X_test_list,
        axis=0
    )

    y_test = np.concatenate(
        y_test_list,
        axis=0
    )

    overall_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    # ---------------------------------------------------------
    # 8. Print final centralized metrics
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CENTRALIZED BASELINE RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {overall_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {overall_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {overall_metrics['recall']:.4f}"
    )

    print(
        f"F1       : {overall_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {overall_metrics['auc']:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()