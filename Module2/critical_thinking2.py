import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys

def load_data(file_path):
    return pd.read_csv(file_path)

def train_classifier(data):
    # Assuming the last column is the target
    X = data.iloc[:, :-1]  # Features: all columns except the last
    y = data.iloc[:, -1]   # Target: the last column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return knn

def predict_iris(knn, features):
    prediction = knn.predict([features])
    return prediction[0]

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage Expected: python critical_thinking2.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        sys.exit(1)

    # Parse input features
    try:
        sepal_length = float(sys.argv[1])
        sepal_width = float(sys.argv[2])
        petal_length = float(sys.argv[3])
        petal_width = float(sys.argv[4])
    except ValueError:
        print("All inputs must be floating point numbers.")
        sys.exit(1)

    features = [sepal_length, sepal_width, petal_length, petal_width]

    # Load data
    data = load_data('data.csv')

    # Train classifier
    knn = train_classifier(data)

    # Make prediction
    iris_type = predict_iris(knn, features)
    print(f'Predicted Iris type: {iris_type}')
