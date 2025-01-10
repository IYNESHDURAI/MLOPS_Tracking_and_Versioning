import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to log experiment details
def run_experiment(n_estimators):
    with mlflow.start_run():
        # Initialize the model
        model = RandomForestClassifier(n_estimators=n_estimators)

        # Fit model
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "model")
        return accuracy

# Experiment: Try different values of n_estimators
for n_estimators in [10, 50, 100]:
    accuracy = run_experiment(n_estimators)
    print(f"Model with {n_estimators} estimators accuracy: {accuracy}")