# Diabetes Prediction

This project aims to predict the onset of diabetes in patients using various machine learning models. The dataset is sourced from the US National Institute of Diabetes and Digestive and Kidney Diseases and includes features such as glucose levels, blood pressure, insulin levels, and body mass index (BMI). The project involves data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and deployment through a simple Flask web application.

## Key Features

1. **Data Preprocessing**:
    - Loading and normalizing the dataset.
    - Handling missing values and scaling numerical features.

2. **Exploratory Data Analysis (EDA)**:
    - Visualizing data distributions and relationships using pair plots.
    - Creating a correlation heatmap to understand feature correlations.

3. **Model Training**:
    - Implementing Logistic Regression, Random Forest, and Support Vector Machine (SVM) models.
    - Splitting the dataset into training and testing sets.
    - Evaluating model performance using accuracy, classification reports, and confusion matrices.

4. **Hyperparameter Tuning**:
    - Using GridSearchCV to tune hyperparameters for the Random Forest model.
    - Selecting the best model based on cross-validation accuracy.

5. **Model Evaluation**:
    - Plotting ROC curves to assess model performance.
    - Computing and visualizing the Area Under the Curve (AUC) for the ROC.

6. **Model Deployment**:
    - Saving the trained model using Joblib.
    - Creating a Flask web application to serve the model for making predictions.

## License

This project is licensed under the MIT License.

