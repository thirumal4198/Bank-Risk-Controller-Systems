# Bank-Risk-Controller-Systems
Loan Default Prediction Project
This project aims to predict loan defaults using machine learning models. The main goal is to build a Streamlit application that showcases data, performs exploratory data analysis (EDA), and makes predictions based on user inputs.

Project Overview
The project involves the following steps:

Data Loading and Preprocessing
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Building a Streamlit Application for Prediction
Data
The dataset used in this project contains information about loan applicants. It includes features such as contract type, gender, income type, occupation type, and more.

Installation
To run this project, you'll need to have Python installed on your machine along with the following libraries:

pandas
numpy
scikit-learn
imbalanced-learn
plotly
seaborn
streamlit
You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn imbalanced-learn plotly seaborn streamlit
Running the Application
To run the Streamlit application, navigate to the project directory and execute the following command:

bash
Copy code
streamlit run project.py
Project Structure
bash
Copy code
.
├── cleaned_data.csv              # The cleaned dataset used for model training and EDA
├── project.py                    # Main script for the Streamlit application
├── DecisionTreeClassifier_model.pkl  # Trained Decision Tree model saved as a pickle file
├── scaler.pkl                    # Trained scaler object saved as a pickle file
├── README.md                     # This README file
└── requirements.txt              # List of required packages
Streamlit Application
The application has three main sections accessible via the sidebar:

Data: Displays the dataset and model performance metrics.
EDA - Visual: Shows various plots for exploratory data analysis, including distributions of numerical and categorical features.
Prediction: Allows users to input feature values and predict whether a customer will default on their loan.
Model Performance Metrics
The following models were trained and evaluated:

Model	ROC AUC	Precision	Recall	F1 Score	Accuracy	Log Loss
Decision Tree	0.9755	0.9449	0.9562	0.9505	0.9914	0.3094
Logistic Regression	0.6862	0.6786	0.0008	0.0016	0.9138	0.2756
Random Forest Classifier	0.9986	1.0000	0.9481	0.9734	0.9955	0.0237
Gradient Boosting Classifier	0.7120	0.7023	0.0038	0.0075	0.9140	0.2699
Confusion Matrix Values
Model	Confusion Matrix
Decision Tree	[[257003, 1359], [1068, 23311]]
Logistic Regression	[[258353, 9], [24360, 19]]
Random Forest Classifier	[[258362, 0], [1265, 23114]]
Gradient Boosting Classifier	[[258323, 39], [24287, 92]]
Avoiding Overfitting
To mitigate overfitting, techniques such as cross-validation, model pruning, regularization, and ensemble methods were used. Random oversampling and more sophisticated techniques like SMOTE were also employed to balance the dataset.

Future Improvements
Implement additional models and compare their performance.
Enhance the Streamlit application with more interactive features.
Optimize model training and prediction times.
License
This project is licensed under the MIT License. See the LICENSE file for details.
