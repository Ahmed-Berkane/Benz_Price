# Mercedes-Benz Price Prediction


This project aims to predict the prices of Mercedes-Benz cars in the USA based on various car attributes. Using machine learning, we explored multiple models to find the one that best predicts car prices based on a range of features. Additionally, the project includes a Flask web application to provide predictions through a user-friendly interface.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling](#modeling)
    - [Model Selection](#model-selection)
5. [Flask App](#flask-app)
6. [Setup](#setup)

---

### 1. Project Overview
   - Objective: Predict car prices based on various features.
   - Key Features: Year of manufacture, mileage, engine size, and more.
   - Outcome: Selected Linear Regression model for its high Adjusted R-squared score of 0.92.

### 2. Data Source
   - Dataset: [USA Mercedes Benz Prices Dataset](https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset?select=usa_mercedes_benz_prices.csv)
   - Size: Contains various car attributes including price, model, year, and mileage.

### 3. Exploratory Data Analysis (EDA)
   - Analysis included:
      - Data cleaning and preprocessing.
      - Visualizations (heatmaps, scatter plots) to identify correlations.
      - Feature importance analysis to assess the impact of attributes on car price.

### 4. Modeling
   - Five different models were tested:
      - Linear Regression
      - Elastic net Regression
      - Random Forest
      - Gradient Boosting
      - XGBoost
   - Model Comparison:
      - Evaluated each model based on R-squared and Adjusted R-squared.
      - Linear Regression achieved the highest Adjusted R-squared score of 0.92.

#### Model Selection
   - Final Model: **Linear Regression**
      - Rationale: Linear Regression was chosen for its simplicity and interpretability, as well as its superior Adjusted R-squared score compared to other models.

### 5. Flask App
   - A Flask web application is included for local predictions.
   - Features:
      - Input car attributes via a form.
      - Provides predicted price based on user inputs.
   - Note: The app currently runs locally, with deployment planned for future iterations.

### 6. Setup
   - **Clone the repository**:
     ```bash
     git clone https://github.com/Ahmed-Berkane/Benz_Price.git
     ```
   - **Create a virtual environment**:
     ```bash
     python3 -m venv venv

     # On Windows:
     .\venv\Scripts\activate 

     # On macOS:  
     source venv/bin/activate
     ```
   - **Install dependencies**:
     ```bash
     cd Benz_Price
     pip install -r requirements.txt
     ```
   - **Run the Flask app locally**:
     ```bash
     python application.py
     ```
   - Once the app is running, open your browser and go to http://127.0.0.1:5000/predictdata to see the app.

   - **Additional Notes**:
      - Ensure all required libraries are installed as per `requirements.txt`.
      - Data files should be placed in the specified directories if not included.



