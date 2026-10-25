from dataset import load_hospital_partitions
from model import StrokeMLP
from trainer import train_model
from evaluate import evaluate_model

partitions = load_hospital_partitions(
    num_hospitals=4
)

hospital = partitions[0]

X_train = hospital["X_train"]
y_train = hospital["y_train"]

X_test = hospital["X_test"]
y_test = hospital["y_test"]

class_weights = hospital["class_weights"]

input_size = X_train.shape[1]

model = StrokeMLP(input_size)

model, loss = train_model(
    model,
    X_train,
    y_train,
    class_weights,
    epochs=5
)

metrics = evaluate_model(
    model,
    X_test,
    y_test
)

print("\nFinal Metrics")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")