# NLP-Twitter-Sentiment-Classification

# 🧠 COVID-19 Twitter Sentiment Analysis (NLP Project)

This project applies Natural Language Processing (NLP) techniques to classify the sentiment of tweets related to the COVID-19 pandemic. The goal is to accurately identify one of five sentiment classes from raw tweet text using supervised machine learning models.

---

## 📂 Dataset

The dataset consists of tweets related to COVID-19 and includes the following columns:

- `OriginalTweet`: The tweet text.
- `Sentiment`: The labeled sentiment class:
  - `0`: Negative
  - `1`: Positive
  - `2`: Neutral
  - `3`: Extremely Positive
  - `4`: Extremely Negative

| Sentiment Class        | Samples |
|------------------------|---------|
| Positive               | 11,422  |
| Negative               | 9,917   |
| Neutral                | 7,713   |
| Extremely Positive     | 6,624   |
| Extremely Negative     | 5,481   |

---

## 🔍 Problem Statement

To classify tweets into **one of five sentiment categories** using machine learning and NLP techniques. This is a multi-class text classification task.

---

## 🛠️ Preprocessing Steps

1. **Text Cleaning**:
   - Removed URLs, mentions, hashtags, emojis using Regex.
   - Lowercased text.
2. **Stopwords**: Stop words are filtered out (such as “the”, “a”, “an”, “in”) because they carry little value.
3. **Text Normalization**:
Here we have two methods, lemmatization and stemming. Their goal is to reduce words to a base/root form so that variations of a word are treated the same during analysis. 

    In my case I will choose lemmatization because:

    - Preserves word meaning (e.g. "better" → "good", "running" → "run")

    - Respects part of speech, which matters for sentiment

    - Keeps the sentence semantically intact

    Stemmers are more aggresive and thus lemmatizers are prefered.
4. **POS Tagging and Tokenization**: 
    - Lemmatizers need to know the role of a word in a sentence to convert it to the correct base form.

    - POS tagging tells the lemmatizer the correct "type" of word, so it can choose the right lemma.

5. **POS Filter**:

    - According to theory not all part of speech are importan to sentiment analysis. 
    - There is no universal guidance what we should keep and what not but as a rule of thump from linguistics nouns , adjectives and adverbs include the most meaningfull information on sentiment analysis.

    - Created a function to filter only tokens of our interest regarding their POS classification.

##  🤖 Classification
1. **Label Encoding**:
   - Sentiments were mapped to numeric labels (0–4).
2. **Train-Test-X-y Split** 
3. **TF-IDF Vectorizer**
    - Algorithms do not understand words so we need to transform our text into a numeric representation. 
4. **Models used**:

| Model                  | Accuracy                              |
|------------------------|--------------------------------------|
| Logistic Regression    | 0.48 |
| Random Forest          |  0.46 (slow)|
| Multinomial Naive Bayes| 0.35                    |
| Linear SVM             | 0.47
| Rbf SVM                | 0.52 (took 19 minutes -- very slow)

---

## 📊 Results

For a 5-class sentiment classification problem on tweets, the current results (around 48% accuracy, f1-scores in the 0.4–0.5 range) are not unusual since we are using classical ML algorithms.

- The dataset has overlapping classes (like Positive vs. Extremely Positive).

- We are using TF-IDF, which ignores context and word order.
