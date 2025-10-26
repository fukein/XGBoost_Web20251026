# XGBoost_Web20251026

Clinical Disease Risk Prediction Web App
A Streamlit-based interactive web application that predicts clinical disease risk using a pre-trained XGBoost model, with SHAP (SHapley Additive exPlanations) visualization to interpret feature contributions.

## Key Features
1.Clinical Indicator Input: Intuitive 3-column layout for entering 21 clinical features (e.g., age, BMI, blood pressure, blood glucose).
2.Real-Time Risk Prediction: Outputs disease risk level (Low/Medium/High) and corresponding probability (%).
3.Model Interpretability: Visualizes top 10 feature contributions via SHAP waterfall plots (red = risk-increasing, blue = risk-decreasing).
4.User-Friendly Interface: Custom CSS styling, clear result display, and support for viewing full SHAP value details.

## Core Dependencies
Ensure the following packages are installed before running:  pip install streamlit xgboost shap pandas numpy matplotlib joblib

## How to Run
1.Place your pre-trained XGBoost model file (named xgboost_model.pkl) in the project root directory.
2.Run the application with the command: streamlit run app.py
3.Enter clinical indicators in the input area and click "Predict Risk" to get results.

## Notes
1.The model file xgboost_model.pkl is required; the app will terminate with an error if the file is missing.
2.All input features are restricted to clinically reasonable ranges to ensure data validity.
3.SHAP plots are generated in real-time after prediction, showing the most influential indicators for risk assessment.
