from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    random_state=42
)

model = RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(X_train, y_train)
print("Validation Accuracy: ", model.score(
    X_test,y_test
))

joblib.dump(model , "iris_model.pkl")
print("Model saved to iris_model.pkl")