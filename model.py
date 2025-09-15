import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('C:/Users/keltron/Desktop/sentiment-analysis/Reviews.csv')
df = df[['Text', 'Score']].dropna()

# Remove neutral reviews (score == 3)
df = df[df['Score'] != 3]

# Label sentiment
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')

# Downsample to balance dataset
positive = df[df['Sentiment'] == 'positive']
negative = df[df['Sentiment'] == 'negative']
positive_downsampled = resample(positive, replace=False, n_samples=len(negative), random_state=42)

df_balanced = pd.concat([positive_downsampled, negative]).sample(frac=1, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df_balanced['Text'])
y = df_balanced['Sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully.")
