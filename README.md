# Introduction
Financial fraud, particularly in online transactions, has become a critical issue with the rise of digital banking and e-commerce. Detecting fraudulent transactions in real time is essential to prevent monetary loss and maintain trust in financial systems.
This project utilizes the Fraud Detection Dataset from Kaggle and implements a machine learning-based prediction pipeline integrated into an interactive Streamlit web application.
It focuses on building a machine learning-based system to detect fraudulent transactions in financial datasets. The workflow covers all key stages of the data science process, including data preprocessing, exploratory data analysis (EDA), model development, and evaluation. This project focuses on detecting fraudulent transactions using machine learning.
The dataset contains details about financial transactions and labels indicating whether they are fraudulent.
The primary goal is to build a predictive model that can identify potential fraud while handling class imbalance.

# Dataset Overview

Datase Link: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download

This dataset comes from Kaggle and includes approximately 6.3 million records of financial transactions.
Type of Data: Tabular transaction data

Key Features:
type – Transaction category (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT)

amount – Transaction amount

oldbalanceOrg – Account balance before transaction (Sender)

newbalanceOrig – Account balance after transaction (Sender)

oldbalanceDest – Account balance before transaction (Receiver)

newbalanceDest – Account balance after transaction (Receiver)

isFraud – Target variable (1 = Fraudulent, 0 = Genuine)

The dataset is highly imbalanced, with fraudulent transactions forming a very small percentage of total records.

## Data Exploration
The dataset was loaded and inspected for structure, missing values, and class distribution.
Key steps included:
- Viewing the first few rows with `head()`
- Checking column data types and non-null counts using `info()`
- Counting the number of fraudulent vs. non-fraudulent transactions
- Checking for missing values
- Determining the total number of rows in the dataset

## Data Preprocessing
To prepare the dataset for model training, preprocessing was applied to handle categorical and numerical features:
- Numerical variables were standardized using `StandardScaler`
- A `ColumnTransformer` was used to combine these steps into a single transformation pipeline
- Data Cleaning – Removed irrelevant columns, ensured numeric type conversions.
- Categorical Encoding – Converted type into numerical form using one-hot encoding using `OneHotEncoder`
- Handling Imbalance – Applied techniques such as oversampling or SMOTE to balance the dataset.
- Feature Scaling – Standardized numerical columns to improve model performance.

This ensured that all features were in a suitable format for model training.

## Model Building
The model chosen for this task was **Logistic Regression**, due to its interpretability and effectiveness in binary classification.
Parameters used:
- `class_weight='balanced'` to address the imbalance between fraudulent and non-fraudulent transactions
- `max_iter=1000` to ensure convergence during training

The preprocessing steps and the classifier were combined into a single `Pipeline` for streamlined training.

A fraud detection pipeline was built, saved as fraud_detection_pipeline.pkl, and later loaded into the Streamlit app for predictions.

Modeling Steps:

Train-Test Split: 80% training, 20% testing

Algorithms Tested:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting / XGBoost

Evaluation Metrics:
- Precision, Recall, F1-score (Recall is prioritized to catch maximum frauds)
- ROC-AUC score

Best Model: Random Forest Classifier (tuned hyperparameters) gave a good trade-off between recall and precision.

## Streamlit Prediction App
The Streamlit app serves as an easy-to-use fraud detection tool.
Key Functionalities:

User Inputs:
- Transaction Type
- Amount
- Sender & Receiver Balances (Before and After Transaction)

Real-Time Prediction:
- Outputs 0 (Not Fraud) or 1 (Fraud)
- Displays a green success message for genuine transactions and a red warning for potential fraud.

## Results & Evaluation
After training, the model was evaluated on the test set using:
- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: To visualize true positives, false positives, true negatives, and false negatives
These metrics help determine the effectiveness of the fraud detection system, with particular focus on **Recall** for the fraud class.

High Accuracy in detecting fraudulent transactions with minimal false negatives.

The system can be used for real-time fraud prevention in banking or payment platforms.

Model Deployment through Streamlit allows non-technical users to interact with the prediction system easily.

## Conclusion
This project successfully demonstrates:

- The use of a large-scale financial transaction dataset for machine learning fraud detection.
- The creation of an end-to-end pipeline from data preprocessing to web-based prediction deployment.
- The importance of handling imbalanced datasets and prioritizing recall in fraud detection systems.
