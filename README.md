# Sentiment Analysis with Twitter Data

## Overview
This project focuses on sentiment analysis using Twitter data. The goal is to classify tweets into positive, negative, or neutral sentiments using natural language processing techniques.

## Dataset
The dataset used in this project is derived from the Sentiment140 dataset, as described in the paper ["Twitter Sentiment Classification using Distant Supervision"](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) by Alec Go, Richa Bhayani, and Lei Huang. The dataset contains 1,600,000 tweets annotated with polarity (0 = negative, 2 = neutral, 4 = positive).

## Task
The tasks completed in this project include:

1. **Data Exploration and Preprocessing:**
   - Tokenization, stemming, and removal of stopwords were performed.
   - Data visualization techniques such as word clouds and histograms were used to gain insights into the dataset.

2. **Building a Bag of Words (BOW) Model:**
   - Implemented BOW model and trained three different classifiers: KNN, Decision Tree, and SVM.

3. **Model Evaluation:**
   - Evaluated the performance of the classifiers using various metrics including confusion matrix, accuracy, and classification report.

4. **Word Embeddings and CNN Model:**
   - Utilized word embeddings (e.g., word2vec, Glove, fasText) and built a Convolutional Neural Network (CNN) model.
   - Compared the performance of the CNN model with the BOW approach.

## Instructions
To run the code and reproduce the results:

1. Clone the repository.
2. Download the sentiment140 dataset from [this link](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) and extract the files.
3. Follow the instructions in the README to set up the environment and install the necessary dependencies.
4. Run the provided scripts to preprocess the data, build models, and evaluate performance.

## Requirements
- Python 3.x
- Libraries such as NLTK, scikit-learn, TensorFlow, Keras, etc.

## Results
Detailed results of model evaluation and performance comparison are provided in the Code.ipynb file.

## Contributors
- Abed Bakkour

## References
- Alec Go, Richa Bhayani, and Lei Huang. "Twitter Sentiment Classification using Distant Supervision." Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1. Association for Computational Linguistics, 2009.