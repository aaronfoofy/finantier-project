# finantier-project
Default prediction of telco customers
READ ME FOR FINANTIER PROJECT -- AARON FOO

OVERVIEW OF FILES
########################################################################################################################################

The Finantier Project.ipynb jupyter notebook has an overview on the various models and types of feature engineering attempted 
before settling for our final model.

The Finantier_Project_API_TEST.ipynb jupyter notebook contains the chosen model (LOGREG + OHE) and the training steps 
undertaken for the default classification model.

From this, encoder.pkl & model.pkl were created. They are the 'pickled' versions of the fitted OneHotEncoder 
and trained logreg model for use in the python script for API implementation.

main.py contains the main bulk of the API implementation for the model, including the customerinfo class definition used to collect customer
data to be fed to the model. 
Unfortunately due to time constraints the API does not directly get data inputs from user, we have to use the
Swagger UI to interact with our model eg: if end point is localhost:8000, use localhost:8000/docs to access the swagger UI and
interact with the API.

To run the programme locally, enter command: 'uvicorn main:app --reload' on the python terminal set to the correct path.
eg:PS C:\Users\aaroncode\PycharmProjects\Financier>uvicorn main:app --reload

Dockerfile contains the commands to build a working docker image of the default classifier application described by main.py

RATIONALE FOR CODE in Finantier_Project_API
########################################################################################################################################
Firstly the code had to be cleaned, after spending some time to understand the data and the type of erronous data present 
simply dropping all the nan values was deemed to be acceptable as we would only lose 5 errornous data points.
*The column of totalCharges had to be dealt with differently where inputs with spaces were instead converted to '0's, an assumption that the
user simply left it blank since they had no totalCharges.

Next using some basic domain knowledge, we dropped unneccessary columns such as customerID and Payment Method.

Recognising that certain categories have only 2 options, convert them to binary digits (technically using OHE would also be ok, but 
limiting no of columns and hence dimensionality would be preferred)
These catergories are: Default (our y value), gender,partner,dependents and phoneservice.

Moving on to handle the rest of the catergorical data, we import category_encoders and utilise their OneHotEncoder to convert the rest 
of the columns to numerical values.
*Important to fit the encoder and pickle it for use in the python script later.

SMOTE: synthetic minority oversampling technique was considered for this project and was run but found to not yield significant improvements
in the results. This technique aims to balance imbalanced datasets through creation of synthetic points. 


Split the data into train and test set

Using sklearns scaler, we scale our features in order to ensure they all contribute evenly when training our model.

Then using sklearns logistic regression model, we fit the model according to the training data and proceed to test it against our test data.

I used a general .score() to measure the general accuracy of our model (correct predictions/ total), granted in this use case minimising
the number of false negatives has more weight in order to avoid taking on customers which are likely to default.

In order to have a visual gauge of the model performance, I printed out the heatmap of the model as well as the precision-recall curve.
This curve is used to for binary classification models (such as ours) with imbalanced datasets, the number of defaults is significantly 
lower than the number of non-defaults and thus I found it appropriate to use such a curve.
This curve can help future users understand the tradeoff between precision and recall the model would undergo when prioritising
an objective.

Finally we pickle our model to use in the pycharm files.


DOCKER
########################################################################################################################################
The Dockerfile contained within the pycharm files aims to create a docker image which will run the API instructions conttained within
main.py with the endpoint set as localhost.
Note that when running the dockerimage, remeber to set the ports and container name in order to find the container and subsequently 
open it in browser while it is running.

After typing localhost:8000 in your browser a "Hello, stranger" message will appear. In order to interact with our model,
we have to add a /docs to the end of the url where we will reach the swagger UI to visualise our GETs and POSTs in main.py.

POST/predict will allow one to interact with the model by entering appropriate responses for the matching entries eg: "gender" : "Male".
