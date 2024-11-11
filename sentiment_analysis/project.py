import re
import random
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load Dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Split data into reviews and labels
reviews = [" ".join(words) for words, label in documents]
labels = [label for _, label in documents]

# Step 2: Preprocess the Text Data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    words = text.split()  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

# Apply preprocessing
cleaned_reviews = [preprocess_text(review) for review in reviews]

# Step 3: Convert Text to Numerical Data (TF-IDF Vectorization)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(cleaned_reviews).toarray()

# Step 4: Encode the Labels
y = [1 if label == 'pos' else 0 for label in labels]

# Step 5: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Choose and Train a Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Test on New Reviews
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = tfidf.transform([processed_review])
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction > 0 else "Negative"

print(predict_sentiment(input("Enter rating: ")))
print(predict_sentiment(input("Enter rating: ")))
