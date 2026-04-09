# Twitter Sentiment Analysis Project

## Project Overview
This project implements a machine learning pipeline to classify the sentiment of tweets into three categories: Negative, Neutral, and Positive. It uses a Logistic Regression model trained on TF-IDF features to identify emotional patterns in social media text.

## Pipeline Workflow

### 1. Data Cleaning and Preprocessing
* The dataset is loaded using 'cp1252' encoding to handle special characters.
* A custom 'clean' function converts text to lowercase and removes URLs and handles.
* Special characters and numbers are stripped, leaving only alphabetic text.
* Empty strings are converted to NaN and dropped to ensure data integrity.

### 2. Feature Engineering
* Text is converted to numerical data using 'TfidfVectorizer'.
* The model uses both unigrams and bigrams (ngram_range=(1,2)) to capture context.
* Words appearing in more than 90% or fewer than 4 documents are filtered out.

### 3. Model Training
* A Logistic Regression model is trained with a 'multinomial' configuration.
* The 'class_weight=balanced' parameter is used to handle uneven class distributions.
* Data is split into 80% training and 20% testing sets with stratification.

## Results and Insights

### Performance Metrics
* Accuracy Score: 64.21%.
* Positive Class: Highest F1-score (0.69).
* Neutral Class: Hardest to predict (F1-score 0.60) due to overlap with other categories.

### Word Importance
The model assigns weights to specific words to determine sentiment:
* Negative: sick, tired, poor, sucks, stupid, bad, hate, sad.
* Neutral: google, office, live, guess, eating, vegas, lunch.
* Positive: happy, cool, awesome, amazing, great, good, nice, love.

### Manual Testing
* "I am so frustrated with this service, it is terrible" -> NEGATIVE (47.6% confidence).
* "Today is a standard Monday morning" -> NEUTRAL (46.1% confidence).

## Deployment
* The final model and vectorizer are saved as '.pkl' files for use in production environments.
