# Customer Churn Prediction

A **Machine Learning project** to predict customer churn using a trained model built on a Kaggle dataset, with a Streamlit web app for interactive predictions and a Google Colab notebook for experimentation.

---

## Project Description

This project predicts whether a customer is likely to churn using machine learning.  

- **Streamlit app (`app.py`)**: Enter customer details and get instant churn predictions.  
- **Trained model (`churn_model.pkl`)** and **scaler (`scaler.pkl`)**: Built using a Kaggle dataset and used for predictions.  
- **Google Colab notebook (`customer_churn_prediction.py`)**: Contains the step-by-step workflow and exploration of the model.

---

##  Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application to interactively predict churn |
| `churn_model.pkl` | Trained machine learning model |
| `scaler.pkl` | Feature scaler used by the model |
| `customer_churn_prediction.py` | Google Colab notebook with analysis, EDA, and model training |
| `.gitattributes` | Used by Git LFS to handle large files (`.pkl`) |

---

##  How to Run Streamlit App

1. Make sure Python is installed (Python 3.8+ recommended).  
2. Install required packages:

    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```
3. Run the app:

    ```bash
    streamlit run app.py
    ```
