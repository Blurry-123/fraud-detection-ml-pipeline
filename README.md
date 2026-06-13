# Fraud Detection Pipeline Using Machine Learning

## Overview

This project develops a supervised machine learning pipeline to identify fraudulent transactions in a highly imbalanced financial dataset. The objective is to build an effective fraud detection system using data preprocessing, class imbalance handling techniques, and classification algorithms.

Since fraudulent transactions represent only a small percentage of the total transactions, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the dataset before training machine learning models.

The project compares Logistic Regression and Random Forest classifiers and evaluates their performance using Precision, Recall, and ROC-AUC metrics instead of Accuracy alone.

---

## Project Objectives

* Detect fraudulent transactions using supervised machine learning.
* Handle severe class imbalance using SMOTE.
* Train and evaluate multiple classification algorithms.
* Compare model performance using Precision, Recall, and ROC-AUC.
* Identify the most important factors contributing to fraud detection.

---

## Dataset Information

* Dataset: Base.csv
* Total Records: 1,000,000
* Features: 31
* Target Variable: fraud_bool

  * 0 = Legitimate Transaction
  * 1 = Fraudulent Transaction

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Matplotlib
* Jupyter Notebook

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Data loading and inspection
* Missing value analysis
* Feature and target separation
* Categorical variable encoding using One-Hot Encoding

### 2. Handling Class Imbalance

* Applied SMOTE (Synthetic Minority Over-sampling Technique)
* Balanced minority fraud class in the training dataset

### 3. Model Development

#### Logistic Regression

* Baseline classification model

#### Random Forest Classifier

* Ensemble learning algorithm
* Tuned for improved fraud detection performance

### 4. Model Evaluation

Models were evaluated using:

* Precision
* Recall
* ROC-AUC Score
* Classification Report
* Confusion Matrix

---

## Results

| Model               | Precision | Recall | ROC-AUC |
| ------------------- | --------: | -----: | ------: |
| Logistic Regression |     0.023 |  0.611 |   0.745 |
| Random Forest       |     0.061 |  0.279 |   0.813 |

### Final Model Performance (Random Forest)

* Accuracy: 97.05%
* Precision: 6.11%
* Recall: 27.88%
* ROC-AUC: 81.26%

---

## Feature Importance Analysis

Top features influencing fraud detection:

1. device_os_windows
2. keep_alive_session
3. customer_age
4. phone_home_valid
5. payment_type_AC
6. has_other_cards
7. current_address_months_count
8. proposed_credit_limit
9. income
10. bank_months_count

These results indicate that behavioral, device-related, and customer verification features play a significant role in identifying fraudulent transactions.

---

## Conclusion

A complete fraud detection pipeline was successfully developed using machine learning techniques. SMOTE was used to address class imbalance, and both Logistic Regression and Random Forest models were evaluated. The Random Forest classifier achieved the best overall performance with a ROC-AUC score of 0.813 and an accuracy of 97.05%, making it the preferred model for fraud detection.

This project demonstrates practical applications of supervised learning, imbalanced data handling, feature engineering, model evaluation, and fraud analytics.

---

## Repository Structure



├── Fraud_Detection_Pipeline.ipynb

├── README.md


---

## Author

Shreya Acharya

Aspiring Data Scientist | Machine Learning Enthusiast | IBM Data Science Professional Certificate
