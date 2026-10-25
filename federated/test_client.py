from federated.dataset import load_hospital_partitions
from federated.client import StrokeClient


# Load hospital data
partitions = load_hospital_partitions(
    num_hospitals=4
)

# Use Hospital 0
hospital = partitions[0]

# Create client
client = StrokeClient(
    X_train=hospital["X_train"],
    y_train=hospital["y_train"],
    X_test=hospital["X_test"],
    y_test=hospital["y_test"],
    class_weights=hospital["class_weights"],
    local_epochs=1
)

# Get initial parameters
parameters = client.get_parameters({})

print("Number of parameter arrays:", len(parameters))

for i, param in enumerate(parameters):
    print(
        f"Parameter {i}: shape={param.shape}"
    )

# Train locally
updated_parameters, num_samples, metrics = client.fit(
    parameters,
    {}
)

print("\nLOCAL TRAINING RESULT")
print("---------------------")
print("Training samples:", num_samples)
print("Training loss:", metrics["loss"])

# Evaluate
loss, test_samples, evaluation_metrics = client.evaluate(
    updated_parameters,
    {}
)

print("\nLOCAL EVALUATION RESULT")
print("-----------------------")
print("Test samples:", test_samples)
print("Loss:", loss)

for key, value in evaluation_metrics.items():
    print(f"{key}: {value:.4f}")