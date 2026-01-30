import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset/fake_job_postings.csv")

# Fix target column
data['fraudulent'] = data['fraudulent'].apply(lambda x: 1 if x >= 0.5 else 0)

# Split data
X = data.drop("fraudulent", axis=1)
y = data["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Function to predict one job
def predict_job():
    sample = X_test.sample(1)
    prediction = model.predict(sample)
    return "Fake Job" if prediction[0] == 1 else "Real Job"
