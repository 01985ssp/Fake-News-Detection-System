## Fake New Detection System project to check whether the given news in the social media, news channels,google,or anywhere is real or fake.
import pandas as pd
import numpy as np
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------
# Load Dataset Automatically
# -------------------------
url = "https://raw.githubusercontent.com/rowanz/ai2_reasoning_challenge/master/data/fake_news.csv"

try:
    data = pd.read_csv(url)
except:
    print("Dataset link failed. Using backup sample dataset.")
    data = pd.DataFrame({
        "text": [
            "Government announces new policy",
            "Aliens landed on earth today",
            "Stock market rises significantly",
            "Earth is proven flat by scientists"
        ],
        "label": [1, 0, 1, 0]
    })

# Ensure correct columns
data = data[['text','label']]

# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['text'] = data['text'].apply(clean_text)

# -------------------------
# Split Data
# -------------------------
X = data['text']
y = data['label']   # 0 = Fake, 1 = Real

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Train Model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------
# Evaluate Model
# -------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# Prediction Function
# -------------------------
def predict_news(news):
    news = clean_text(news)
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)

    if result[0] == 1:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

# -------------------------
# Test with Input
# -------------------------
sample_news = input("Enter news text: ")
print("Prediction:", predict_news(sample_news))


### OUTPUT for the above code
 ##[nltk_data] Downloading package stopwords to /root/nltk_data...
 # [nltk_data]   Package stopwords is already up-to-date!
 #  Dataset link failed. Using backup sample dataset.
 #  /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
 # _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
 # _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
 #  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
 #  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
 #  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
 # _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
 # Accuracy: 0.0
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       1.0
           1       0.00      0.00      0.00       0.0

    accuracy                           0.00       1.0
   macro avg       0.00      0.00      0.00       1.0
weighted avg       0.00      0.00      0.00       1.0

 # Enter news text: Government announces new tax policy
 # Prediction: REAL NEWS  ### ///
