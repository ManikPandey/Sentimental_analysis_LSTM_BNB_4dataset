
# Sentiment Analysis Project  

This repository contains a sentiment analysis project built using machine learning and deep learning techniques. The project utilizes **LSTM**, **Logistic Regression**, and **Bernoulli Naive Bayes** algorithms on four diverse datasets for sentiment classification. Pre-trained **GloVe word embeddings** are used for feature extraction to enhance the quality of text representation.

---

## Project Overview  

This project focuses on analyzing sentiments from text data across multiple domains, including social media (tweets), movie reviews, e-commerce, and hospitality reviews. The objective is to classify sentiments as **positive** or **negative** while exploring how different algorithms perform on varied datasets.

---

## Algorithms Implemented  

1. **LSTM**: A deep learning model for handling sequential data effectively.  
2. **Logistic Regression**: A widely used linear classification algorithm for binary sentiment analysis.  
3. **Bernoulli Naive Bayes**: A probabilistic algorithm suitable for binary data classification.

---

## Datasets  

1. **Sentiment140 Dataset**  
   - Contains **1,600,000 tweets** extracted using the Twitter API.  
   - Sentiments are labeled as **0 (negative)** and **4 (positive)**.  
   - [Download Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  

2. **IMDB Movie Reviews Dataset**  
   - Contains **50,000 movie reviews** for binary sentiment classification.  
   - Includes **25,000 training** and **25,000 testing** samples.  
   - [Download Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

3. **Amazon Reviews Dataset**  
   - Contains **34,686,770 reviews** from **6,643,669 users** on various products.  
   - A subset of the data includes **1,800,000 training samples** and **200,000 testing samples**.  
   - [Download Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv)  

4. **Yelp Open Dataset**  
   - Contains millions of reviews on hotels, restaurants, and cafes.  
   - Includes **6,685,900 reviews** in JSON file format.  
   - [Download Dataset](https://metatext.io/datasets/yelp-open-dataset)  

---

## Project Structure  

```plaintext
├── glove/                          # Pre-trained GloVe embeddings (50d, 100d, 200d, 300d)
├── kaggle/                         # Stores Kaggle dataset files or configurations
├── sentiment140.xlsx               # Sentiment140 dataset for training
├── logistic_model_with_vectorizer.sav   # Logistic regression model with vectorizer
├── trained_logistic_model.sav      # Trained logistic regression model
├── trained_logistic_model_vectorizer.sav  # Vectorizer for logistic regression
├── trained_model.sav               # Final trained model
├── vectorizer.sav                  # Vectorizer for text preprocessing
├── modelLRBNB.ipynb                # Notebook for Logistic Regression and Naive Bayes training
├── model_predictions.ipynb         # Notebook for predictions and testing
├── glove.6B.50d-300d.txt           # Pre-trained word embeddings


