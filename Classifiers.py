import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load data
df = pd.read_csv('online_shoppers_intention.csv')

# Encode text columns into numbers
le = LabelEncoder()
for col in ['Month', 'VisitorType', 'Weekend']:
    df[col] = le.fit_transform(df[col])

# Split features (X) and target (y)
X = df.drop(columns=['Revenue'])
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- THE SPEED FIX: Scale the numerical columns ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models (Using the fast, scaled data!)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

print("--- Model Accuracy Scores ---")
for name, model in models.items():
    # Train using scaled data for lightning-fast speeds
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    print(f"{name}: {score * 100:.2f}%")
