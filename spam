import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
# Load the dataset
data = pd.read_csv("/content/spam dataset.csv", encoding='latin-1')
print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode the labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text processing
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer(stop_words=stop_words)

# Features and labels
X = data['message']
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear')
}
for name, clf in classifiers.items():
    print(f'\nTraining {name}...')

    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', clf)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate SMS', 'Spam SMS'], yticklabels=['Legitimate SMS', 'Spam SMS'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
# Example usage of the trained model (Naive Bayes)
example_texts = ["Congratulations! You've won a free ticket to Bahamas. Call now!",
                 "Hey, are we still meeting for lunch tomorrow?"]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB())
])
pipeline.fit(X_train, y_train)
example_preds = pipeline.predict(example_texts)

print("\nExample Predictions:")
for text, pred in zip(example_texts, example_preds):
    print(f'Text: {text}\nPrediction: {"Spam" if pred else "Ham"}\n')
