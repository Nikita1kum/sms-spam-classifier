## SMS Spam Classifier
This repository contains a project that classifies SMS messages as either spam or non-spam (ham) using machine learning techniques in Python.

1.Table of Contents
2.Introduction
3.Dataset
4.Requirements
5.Installation
6.Usage
7.Data Preprocessing
8.Modeling
9.Evaluation
10.Visualization


## Introduction
The goal of this project is to build a machine learning model that can accurately classify SMS messages as spam or non-spam. This project demonstrates the entire workflow from data preprocessing to model evaluation and visualization.

## Dataset
The dataset used in this project is the SMS Spam Collection Data Set, which can be downloaded from the UCI Machine Learning Repository.

## Requirements
Python 3.7 or higher
Pandas
NumPy
Scikit-learn
NLTK
Matplotlib
Seaborn
Wordcloud
Installation
Clone the repository:

## Data Preprocessing
Lowercasing
Tokenization
Removing special characters and punctuation
Removing stop words
Stemming


## Modeling
Multiple machine learning models are trained and evaluated:

Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
Decision Tree
Random Forest
k-Nearest Neighbors (k-NN)
Gradient Boosting


## Evaluation
Model performance is evaluated using:

Accuracy
Confusion Matrix
Classification Report
Visualization
Word clouds are generated to visualize the most frequent words in spam and non-spam messages.
