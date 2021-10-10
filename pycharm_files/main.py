#import uvicorn
#import numpy as np
#import category_encoders as ce
#from CustomerInfo import customer_info
from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


class customer_info(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup  :       str
    DeviceProtection  :   str
    TechSupport       :   str
    StreamingTV       :   str
    StreamingMovies   :   str
    Contract          :   str
    PaperlessBilling  :   str
    PaymentMethod     :   str
    MonthlyCharges    :   float
    TotalCharges      :   str


app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)  # Load our logreg model into model object
pickle_in = open("encoder.pkl", "rb") # Load fitted encoder into encoder object
encoder = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, stranger'}
# Return a key-value pair to the homepage

@app.post('/predict')
def predict_default(data:customer_info):
    data_dict = data.dict()
    # Convert dictionary to dataframe
    x = pd.DataFrame([data_dict])
    customerID = x['customerID']
    # Preprocessing
    list = ['customerID', 'PaymentMethod', 'PaperlessBilling']
    x = x.drop(list, axis=1)  # Drop unneccessary fields from features
    # Recognise, gender, Partner, Dependents and PhoneService can also be changed to 0,1 values
    x['gender'] = (x['gender'] == 'Female').astype(int)
    x['Partner'] = (x['Partner'] == 'Yes').astype(int)
    x['Dependents'] = (x['Dependents'] == 'Yes').astype(int)
    x['PhoneService'] = (x['PhoneService'] == 'Yes').astype(int)
    # Change senior citizen & total charges to appropriate types as well
    x['SeniorCitizen'] = x['SeniorCitizen'].astype(int)
    x['TotalCharges'] = x['TotalCharges'].replace(r'^\s+$', 0, regex=True)
    x['TotalCharges'] = pd.to_numeric(x['TotalCharges'])
    x = encoder.transform(x) # applying the fitted encoder onto x
    prediction = model.predict(x)
    if (prediction[0] > 0.5):                # If else on the prediction to print a statement
        prediction = "Customer ID: " + customerID + " is likely to default"
    else:
        prediction = "Customer ID: " + customerID + " is unlikely to default"
    return{
        'prediction':prediction
    }
# uvicorn main:app --reload (run on python terminal)



