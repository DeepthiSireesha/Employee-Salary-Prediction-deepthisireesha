import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODEL_PATH = 'salary_prediction_model.joblib'
DATA_PATH = 'synthetic_employee_data.csv'

# --- Synthetic Data Generation ---
@st.cache_data
def generate_synthetic_data(num_samples=1000):
    """Generates synthetic employee data for salary prediction."""
    np.random.seed(42)

    years_experience = np.random.uniform(0, 20, num_samples).round(1)
    education_levels = np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], 
                                      num_samples, p=[0.15, 0.45, 0.30, 0.10])
    job_titles = np.random.choice(['Software Engineer', 'Data Scientist', 'Project Manager', 
                                 'HR Specialist', 'Sales Representative', 'Marketing Analyst'], num_samples)
    cities = np.random.choice(['New York', 'San Francisco', 'Austin', 'Chicago', 
                             'Boston', 'Seattle', 'Remote'], num_samples)

    # Base salary with some noise
    base_salary = 40000 + (years_experience * 5000)

    # Adjust salary based on education
    salary_education_boost = np.select(
        [education_levels == 'Bachelors', education_levels == 'Masters', education_levels == 'PhD'],
        [15000, 30000, 50000],
        default=0
    )

    # Adjust salary based on job title
    salary_job_boost = np.array([
        {'Software Engineer': 20000, 'Data Scientist': 25000, 'Project Manager': 15000,
         'HR Specialist': 5000, 'Sales Representative': 10000, 'Marketing Analyst': 8000}[jt]
        for jt in job_titles
    ])

    # Adjust salary based on city
    salary_city_boost = np.array([
        {'New York': 20000, 'San Francisco': 25000, 'Austin': 10000,
         'Chicago': 12000, 'Boston': 15000, 'Seattle': 18000, 'Remote': 0}[c]
        for c in cities
    ])

    # Add random noise
    noise = np.random.normal(0, 10000, num_samples)

    salary = base_salary + salary_education_boost + salary_job_boost + salary_city_boost + noise
    salary = np.maximum(salary, 30000).round(-2)  # Ensure non-negative and round to nearest 100

    data = pd.DataFrame({
        'YearsExperience': years_experience,
        'EducationLevel': education_levels,
        'JobTitle': job_titles,
        'City': cities,
        'Salary': salary
    })
    return data

# --- Model Training ---
@st.cache_resource
def train_and_save_model(data):
    """Trains a RandomForestRegressor model and saves it."""
    X = data[['YearsExperience', 'EducationLevel', 'JobTitle', 'City']]
    y = data['Salary']

    # Define preprocessing steps
    numerical_features = ['YearsExperience']
    categorical_features = ['EducationLevel', 'JobTitle', 'City']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline with preprocessor and regressor
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    st.info("Training the model... This might take a moment.")
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, MODEL_PATH)
    st.success("Model trained and saved successfully!")
    return model_pipeline

# --- Load or Train Model ---
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        st.sidebar.info("Loading pre-trained model...")
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Model loaded!")
    else:
        st.sidebar.warning("Model not found. Generating data and training a new model...")
        data = generate_synthetic_data()
        model = train_and_save_model(data)
    return model

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Employee Salary Predictor",
        page_icon="💰",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a more appealing look
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 24px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 6px 12px 0 rgba(0,0,0,0.25);
            transform: translateY(-2px);
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 8px;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 8px;
        }
        .metric-container {
            background-color: #e6f7ff;
            border-left: 5px solid #2196F3;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 1.2em;
            color: #333;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2196F3;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-family: 'Inter', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("💰 Employee Salary Predictor")
    st.markdown("---")

    # Load or train the model
    model = load_or_train_model()

    st.sidebar.header("Employee Details")

    # Input fields
    years_experience = st.sidebar.slider("Years of Experience", 0.0, 30.0, 5.0, 0.5)
    education_level = st.sidebar.selectbox("Education Level", 
                                         ['High School', 'Bachelors', 'Masters', 'PhD'])
    job_title = st.sidebar.selectbox("Job Title", 
                                   ['Software Engineer', 'Data Scientist', 'Project Manager', 
                                    'HR Specialist', 'Sales Representative', 'Marketing Analyst'])
    city = st.sidebar.selectbox("City", 
                              ['New York', 'San Francisco', 'Austin', 'Chicago', 
                               'Boston', 'Seattle', 'Remote'])

    # Prediction button
    if st.sidebar.button("Predict Salary"):
        # Create a DataFrame for the input
        input_data = pd.DataFrame([[years_experience, education_level, job_title, city]],
                                columns=['YearsExperience', 'EducationLevel', 'JobTitle', 'City'])

        try:
            # Make prediction
            predicted_salary = model.predict(input_data)[0]
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Estimated Salary:</div>
                    <div class="metric-value">${predicted_salary:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()  # Visual feedback for prediction
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the model is trained and the inputs are valid.")

    st.markdown("---")
    st.header("Data Distribution Insights")

    # Generate synthetic data for visualization
    synthetic_data = generate_synthetic_data(num_samples=1000)

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Education & Job Distribution", "Salary Analysis", "Data Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Education Levels")
            fig1 = plt.figure(figsize=(8, 5))
            sns.countplot(data=synthetic_data, x='EducationLevel', hue='EducationLevel', 
                         palette='viridis', legend=False)
            plt.title('Distribution by Education Level')
            plt.xlabel('Education')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            plt.close(fig1)
            
        with col2:
            st.subheader("Job Titles")
            fig2 = plt.figure(figsize=(10, 5))
            sns.countplot(data=synthetic_data, x='JobTitle', hue='JobTitle', 
                         palette='magma', legend=False)
            plt.title('Distribution by Job Title')
            plt.xlabel('Job Title')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)

    with tab2:
        st.subheader("Salary by Key Factors")
        
        fig3 = plt.figure(figsize=(12, 6))
        sns.boxplot(data=synthetic_data, x='EducationLevel', y='Salary', 
                   hue='EducationLevel', palette='viridis', legend=False)
        plt.title('Salary Distribution by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Salary ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        plt.close(fig3)
        
        fig4 = plt.figure(figsize=(12, 6))
        sns.boxplot(data=synthetic_data, x='JobTitle', y='Salary', 
                   hue='JobTitle', palette='magma', legend=False)
        plt.title('Salary Distribution by Job Title')
        plt.xlabel('Job Title')
        plt.ylabel('Salary ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig4)
        plt.close(fig4)

    with tab3:
        st.subheader("Summary Statistics")
        st.dataframe(synthetic_data.describe())
        
        st.subheader("Sample Data")
        st.dataframe(synthetic_data.head(10))

    st.markdown("---")
    st.markdown("This is a synthetic data-based salary predictor. Results are for demonstration purposes only.")
    st.markdown("Developed using Streamlit and Scikit-learn.")

if __name__ == "__main__":
    main()
