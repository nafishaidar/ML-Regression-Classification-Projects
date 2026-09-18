import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
titanic = pd.read_csv("titanic.csv")

# Select useful columns
data = titanic[[
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]].copy()

# Fill missing Age
data["Age"] = data["Age"].fillna(data["Age"].median())

# Convert Sex into numbers
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])

# Features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
