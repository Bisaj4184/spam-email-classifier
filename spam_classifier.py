import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "email": [
        "Win money now!!!",
        "Hi, how are you?",
        "Congratulations, you won a prize",
        "Let's meet tomorrow",
        "Free gift cards waiting",
        "Project meeting at 5 PM"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test input
sample = ["Congratulations! You won a free ticket"]
sample_vec = vectorizer.transform(sample)
print("Prediction (1=Spam, 0=Not Spam):", model.predict(sample_vec)[0])
