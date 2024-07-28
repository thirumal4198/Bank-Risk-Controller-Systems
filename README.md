# Loan Default Prediction

## Overview
This project involves developing a machine learning model to predict loan defaults using various classification algorithms. The model is built using a cleaned dataset and includes features for Exploratory Data Analysis (EDA), model training, and prediction through a Streamlit web application.

## Features
   Data Display: View the dataset used for model building and performance metrics.
   
   EDA: Perform and visualize Exploratory Data Analysis on the dataset.
   
   Prediction: Predict whether a customer will default on a loan based on user input.

# Project Structure
      
      ├── data
      │   ├── cleaned_data.csv
      ├── models
      │   ├── DecisionTreeClassifier_model.pkl
      │   ├── LogisticRegression_model.pkl
      │   ├── RandomForestClassifier_model.pkl
      │   ├── GradientBoostingClassifier_model.pkl
      ├── eda
      │   ├── eda_plots.py
      ├── notebooks
      │   ├── EDA.ipynb
      │   ├── Model_Training.ipynb
      ├── app
      │   ├── app.py
      ├── README.md
      ├── requirements.txt

# Installation
## Clone the repository:

      git clone https://github.com/your_username/loan-default-prediction.git
      cd loan-default-prediction
## Create and activate a virtual environment:

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      
## Install the required packages:
      
      pip install -r requirements.txt
      
## Run the Streamlit application:

      streamlit run app.py
## Usage
### Data Display
   View the entire dataset.
   
   View model performance metrics including ROC AUC, Precision, Recall, F1 Score, Accuracy, Log Loss, and Confusion Matrix.
### EDA
   Refresh button to generate and visualize various plots based on numerical, categorical, and binary features.
   
   Numerical Features: Distribution plots and correlation matrix.
   
   Categorical Features: Count plots and bar charts.
   
   Binary Features: Count plots.
   
### Prediction
   Input values for various features to predict loan default.
   
   Display the prediction result and probability of default.

## Model Metrics
      Model	ROC AUC	Precision	Recall	F1 Score	Accuracy	Log Loss	Confusion Matrix
      Decision Tree	0.9755	0.9449	0.9562	0.9505	0.9914	0.3094	[[257003, 1359], [1068, 23311]]
      Logistic Regression	0.6862	0.6786	0.0008	0.0016	0.9138	0.2756	[[258353, 9], [24360, 19]]
      Random Forest Classifier	0.9986	1.0000	0.9481	0.9734	0.9955	0.0237	[[258362, 0], [1265, 23114]]
      Gradient Boosting Classifier	0.7120	0.7023	0.0038	0.0075	0.9140	0.2699	[[258323, 39], [24287, 92]]
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
If you have any questions or suggestions, feel free to reach out:

Email: your_email@example.com
GitHub: your_username
