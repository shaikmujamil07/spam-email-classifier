import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "message": [
        "Win money now",
        "Hello how are you",
        "Free prizes available",
        "Let's meet tomorrow",
        "Claim your reward now"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

# Convert labels to numbers
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
