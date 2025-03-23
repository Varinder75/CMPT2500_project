# 📊 **API Documentation: Predicting with Machine Learning Models**

Welcome to the API documentation! 🚀 This API is designed to interact with machine learning models and predict outcomes based on input data. Whether you're predicting trends in employee data, or just exploring predictions for your business, this API is your tool to get reliable results in no time.

## **Base URL**

All API requests should be made to the below base URL:
http://127.0.0.1:5000


## **Available Endpoints**

### 1. **`/v1/predict1`** - **Predict with Model v1**

**Method**: `POST`  
**Purpose**: This endpoint allows the user to send data to their first machine learning model for a prediction. It will predict the level of pollutants based on the features provided.

#### **Request Format**:

Send your request with the following JSON structure:
```json
{
    "Lag_1": 5.3, 
    "Lag_2": 4.8, 
    "Lag_3": 3.2, 
    "Number of employees": 150
}

#### **Explanation** : The model will process the input and return a prediction (in this case we will get a categorical value- "low", "medium", or "high").
While running the code, we can get some errors like 400, 500. Error "400" means some required field is missing, and error "500" means model or API failure.


### 2. **`/v2/predict2`** - **Predict with Model v2**

**Method**: `POST`  
**Purpose**: This endpoint also allows the user to send data to their first machine learning model for a prediction. It will predict based on the features provided.

#### **Request Format**:

Send your request with the following JSON structure:
```json
{
    "Lag_1": 5.3, 
    "Lag_2": 4.8, 
    "Lag_3": 3.2, 
    "Number of employees": 150
}

#### **Explanation** : The model will process the input and return a prediction (in this case we will get a numerical value). It means the model will tell us the exact amount of pollutants that will be released in the coming years.
While running the code, we can get some errors like 400, 500.

### 3. **Check API Health:**

**Method**: `GET`  
**Purpose**: This endpoint ensures the API is running properly.

#### **Response Format**:

{
    "status": "healthy"
}

#### **Common Issues & Troubleshooting**
-> Invalid Input Data:
Double-check that all fields are correctly formatted and sent in the request body.

-> Server Not Responding:
Ensure the API is running. Try restarting it if necessary.

-> Unexpected Errors:
Check the logs for error details.

### **Key points to remember:**
This API provides predictions based on input data.
Make sure to send valid JSON data to avoid errors.
Use the /health endpoint to check if the API is running.
