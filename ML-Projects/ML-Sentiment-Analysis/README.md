# Twitter Sentiment Analysis

A machine learning pipeline that classifies tweets as **negative**, **neutral** or **positive**. It uses a Logistic Regression model trained on TF-IDF features (unigrams and bigrams) and reaches **64.2% accuracy** on a held-out test set, compared with about 40% for always predicting the majority class.

**Tools:** Python, pandas, scikit-learn, seaborn, matplotlib

## Dataset

* Source: **[DATASET NAME]([https://www.kaggle.com/datasets/OWNER/DATASET-NAME](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset))** on Kaggle.
* The file `tweets.csv` has 4,815 rows and 9 columns. 1,281 rows have no text or label, so **3,533 labeled tweets** are used.
* Only the `text` and `sentiment` columns are used. The other columns (time of tweet, age of user, country, population, land area, density) are ignored.
* Class distribution: neutral 1,429, positive 1,103, negative 1,001.

## Pipeline

### 1. Data cleaning
* A `clean` function lowercases the text and removes URLs, @mentions, numbers and special characters, then normalizes whitespace.
* Tweets that are empty after cleaning are dropped.

### 2. Feature engineering
* `TfidfVectorizer` converts text to numeric features, using unigrams and bigrams (`ngram_range=(1, 2)`).
* Terms that appear in more than 90% of documents or in fewer than 2 documents are removed (`max_df=0.9`, `min_df=2`).
* English stop words are removed and sublinear term-frequency scaling is applied.

### 3. Model training
* Logistic Regression (multinomial, `lbfgs` solver, `C=1.5`, `max_iter=300`).
* `class_weight='balanced'` compensates for the uneven class sizes.
* The data is split 80% / 20% into 2,826 training and 707 test tweets, stratified by label (`random_state=42`).

## Results

All metrics below are measured on the 707 held-out test tweets, using the model's raw predictions.

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Negative | 0.64 | 0.62 | 0.63 |
| Neutral | 0.59 | 0.64 | 0.62 |
| Positive | 0.72 | 0.66 | 0.69 |
| **Overall accuracy** | | | **64.2%** |

* **Baseline:** always predicting the most common class (neutral) would score about 40%, so the model is roughly 24 points better.
* **Positive** is the easiest class (F1 0.69).
* **Neutral** is the weakest (F1 0.62, precision 0.59). Neutral tweets have no strong sentiment words, so they are easily confused with the other two classes.

### Word importance

The highest-weighted words for each class, taken from the model coefficients:

* **Negative:** ugh, tired, poor, sucks, bad, stupid, sorry, hate, miss, sad
* **Neutral:** home, vegas, office, second, guess, live, happened, eating, said, lunch
* **Positive:** wish, cool, amazing, awesome, nice, thank, great, good, thanks, love

The negative and positive words are intuitive. The neutral list is mostly everyday topics and places rather than emotion words, which is what you would expect from a class defined by the absence of strong sentiment.

## Trying the model on new text

The notebook includes `predict_my_tweet(text)`, which cleans a tweet, vectorizes it and returns a prediction with a confidence score (the probability of the model's top class).

### The 0.55 confidence rule (demo heuristic)

By default, if the model's top probability is **below 0.55**, the function reports the tweet as `neutral`. The idea is that when the model is unsure whether a tweet is positive or negative, "neutral" is a reasonable answer. The threshold is the constant `CONFIDENCE_THRESHOLD`, and `predict_my_tweet(text, threshold=0)` returns the raw prediction instead.

This rule is **not part of the evaluated model**. On the test set it changes about 52% of predictions and lowers accuracy from 64.2% to 62.2%, so every metric in this README uses the raw predictions.

| Tweet | Model's top class (confidence) | Reported by `predict_my_tweet` |
|-------|-------------------------------|--------------------------------|
| "I am so frustrated with this service, it is terrible." | negative (58.2%) | negative |
| "Today is a standard Monday morning." | positive (43.9%) | neutral (below 0.55) |
| "i saw my friend today" | negative (50.3%) | neutral (below 0.55) |

The last two rows show the rule in action. The model is not confident about them, and the rule turns them into neutral.

## Model persistence

The trained model and the fitted TF-IDF vectorizer are saved with `joblib` so they can be reused without retraining:

* `model.pkl`: the Logistic Regression classifier
* `tfidf-vectorizer.pkl`: the fitted TF-IDF vectorizer

This project does not include a web app or API.

## Project structure

```
.
├── Tweets.ipynb            # full pipeline: cleaning, training, evaluation
├── tweets.csv              # dataset
├── model.pkl               # trained classifier
├── tfidf-vectorizer.pkl    # fitted vectorizer
├── requirements.txt
└── README.md
```

## How to run

**1. Clone the repository and open the project folder**

```bash
git clone https://github.com/AbdulrahmanGidado/Machine-Learning-Portfolio.git
cd Machine-Learning-Portfolio/ML-Projects/ML-Sentiment-Analysis
```

**2. Create a virtual environment and install the dependencies**

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the notebook**

```bash
jupyter notebook Tweets.ipynb
```

Then choose **Kernel > Restart & Run All**. Running it regenerates `model.pkl` and `tfidf-vectorizer.pkl`.

### Use the saved model without retraining

```python
import re
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf-vectorizer.pkl")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())

tweet = "I love this, it is awesome!"
probs = model.predict_proba(tfidf.transform([clean(tweet)]))[0]
print(model.classes_[probs.argmax()], round(probs.max(), 3))
```

> **Note:** the `.pkl` files were created with scikit-learn 1.6.1, which is why `requirements.txt` pins that version. Only load pickle files from sources you trust.

## Limitations and next steps

* **Dataset size:** only about 3.5K labeled tweets are used, so results may not generalize to other kinds of text.
* **Negation:** the English stop-word list removes words like "not" and "no", so "not good" can lose its meaning. Removing the stop-word filter is worth testing.
* **Vectorizer fitting:** the TF-IDF vectorizer is fitted on the full dataset before the train/test split. Fitting it on the training set only (for example with a scikit-learn `Pipeline`) would be the cleaner approach.
* **One model only:** no other models or cross-validation are compared yet. Naive Bayes, a linear SVM and a transformer such as BERT are natural next steps.
* **Sarcasm and short tweets** are hard for a bag-of-words model, and a proper error analysis of misclassified tweets is still to do.
