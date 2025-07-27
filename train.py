import json
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents data
with open("chatbot/intents.json", "r") as file:
    data = json.load(file)

corpus = []
labels = []

# Extract patterns and tags
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        labels.append(intent["tag"])

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Train classifier
model = LogisticRegression()
model.fit(X, labels)

# Save model, vectorizer, and intents
with open("chatbot/intent_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("✅ Model trained and saved as chatbot/intent_model.pkl")
