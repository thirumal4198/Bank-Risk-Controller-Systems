import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pickle
print(0)
# Load dataset
df = pd.read_csv('cleaned_data.csv')

# Sidebar menu
st.sidebar.title('Navigation')
menu = st.sidebar.radio('Select Menu', ['Data', 'EDA - Visual', 'Prediction'])

# Load the trained Decision Tree model
model_path = 'RFC.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file, encoding='latin1')

#load the encoder 
with open('encoder.pkl','rb') as f:
    encoder = pickle.load(f)


#load the scaler 
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# Load the feature columns used during training
with open('X_train_columns.pkl', 'rb') as file:
    X_train_columns = pickle.load(file)

# Sidebar 1: Data
if menu == 'Data':
    st.title('Dataset')
    #st.write('### Full Dataset')
    

    st.write('### Model Performance Metrics')
    metrics = {
        'Model': ['Decision Tree','Logistic Regression','Random Forest Classifier','Gradient Boosting Classifier'],
        'ROC AUC': [0.9755,0.6862 ,0.9986,0.7120],
        'Precision': [0.9449,0.6786,1.0000,0.7023],
        'Recall': [0.9562,0.0008,0.9481,0.0038],
        'F1 Score': [0.9505,0.0016,0.9734,0.0075],
        'Accuracy': [0.9914,0.9138,0.9955,0.9140],
        'Log Loss': [0.3094,0.2756,0.0237,0.2699],
        'Confusion Matrix': [
            '[[257003, 1359], [1068, 23311]]',
            '[[258353, 9], [ 24360, 19]]',
            '[[258362, 0], [  1265, 23114]]',
            '[[258323, 39], [ 24287, 92]]']
    }
    metrics_df = pd.DataFrame(metrics) 
    st.dataframe(metrics_df)
    st.dataframe(df.head(20))



# Sidebar 2: EDA - Visual
elif menu == 'EDA - Visual':
    st.title('Exploratory Data Analysis')

    if st.button('Refresh EDA'):
        # Numerical columns
        numerical_cols = [
            'EXT_SOURCE_2', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY_x',
            'AMT_GOODS_PRICE_x', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE',
            'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION'
        ]

        # Binary columns
        binary_cols = [
            'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'FLAG_EMAIL',
            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY'
        ]

        # Categorical columns
        categorical_cols = [
            'NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START_x'
        ]

        # Ordinal columns
        ordinal_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']

        # Correlation matrix for numerical features
        st.write('### Correlation Matrix')
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(fig)

        # Helper function to create plots in columns
        def plot_in_columns(cols, title):
            st.write(f'### {title}')
            col1, col2 = st.columns(2)
            for idx, col in enumerate(cols):
                fig_hist = px.histogram(df, x=col)
                fig_box = px.box(df, y=col)
                if idx % 2 == 0:
                    col1.plotly_chart(fig_hist)
                    col1.plotly_chart(fig_box)
                else:
                    col2.plotly_chart(fig_hist)
                    col2.plotly_chart(fig_box)

        # Numerical feature distributions
        plot_in_columns(numerical_cols, 'Distributions of Numerical Features')

        # Categorical feature distributions
        st.write('### Distributions of Categorical Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(categorical_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

        # Binary feature distributions
        st.write('### Distributions of Binary Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(binary_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

        # Ordinal feature distributions
        st.write('### Distributions of Ordinal Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(ordinal_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

# Sidebar 3: Prediction
elif menu == 'Prediction':
    st.title('Loan Default Prediction')

    st.write('Enter the following features to predict whether a customer will default:')
    
    # Binary columns
    binary_cols = [
        'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'FLAG_EMAIL',
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY'
    ]

    # Define columns for input
    col1, col2, col3 = st.columns(3)
    user_input = {}

    # Create a form to capture user input
    with st.form(key='prediction_form'):
        for idx, col in enumerate(df.columns):
            default_value = 0
            if col != 'TARGET':  # Assuming 'TARGET' is the target variable, not an input
                if col in binary_cols:
                    # Handle binary columns with 0 and 1 selection boxes
                    if idx % 3 == 0:
                        with col1:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                    elif idx % 3 == 1:
                        with col2:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                    else:
                        with col3:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                else:
                    # Handle other columns
                    if idx % 3 == 0:
                        with col1:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)
                    elif idx % 3 == 1:
                        with col2:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)
                    else:
                        with col3:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict')



    if submit_button:
        # Convert user input to dataframe
        input_df = pd.DataFrame([user_input])

        # List of categorical columns
        categorical_cols = ['NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START_x']
        
        # Fit and transform the categorical columns
        encoded_cols = encoder.transform(input_df[categorical_cols])
        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
        
        # Drop the original categorical columns
        input_df.drop(categorical_cols, axis=1, inplace=True)
        
        # Concatenate the encoded DataFrame with the original DataFrame
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        columns_to_drop =  ['CNT_FAM_MEMBERS', 'AMT_CREDIT_x']
        input_df = input_df.drop(columns=columns_to_drop)

        # One-hot encode categorical columns
        numerical_cols = input_df.select_dtypes(include=['int64','float64']).columns
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Align input dataframe with training dataframe columns
        missing_cols = set(X_train_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X_train_columns]

        # Predict
        st.dataframe(input_df.head())
        prediction = model.predict(input_df)
        st.write(prediction)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        if prediction[0] == 1:
            st.markdown(f'<div style="background-color:tomato;color:white;padding:10px;">The customer is predicted to default with a probability of {prediction_proba[0]:.2f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:lightgreen;color:black;padding:10px;">The customer is predicted not to default with a probability of {prediction_proba[0]:.2f}.</div>', unsafe_allow_html=True)