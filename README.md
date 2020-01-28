### ML-Classification-Model-Flask-Deployment-WebApp
This is ML Classification project to classify different features of mushroom into two classes - "Edible" or "Poisonous".
The Machine Learning model is deployed on production using Flask API.
Frontend is created using "main.html" to take user inputs via a form and display corresponding class of mushroom.
For each prediction made, historical data is tabulated on the frontend web app and stored as logs in "history.csv" file.

### Prerequisites
You must have Python3, Scikit Learn, Pandas (for Machine Learning Model) and Flask (for API) installed.

### Project Structure
This project has three major parts:
1. Mushroom_Challenge.py - This contains code for our Machine Learning model to predict class of mushroom based on training data in 'mushrooms.csv' file. Only main features are used which were identified during feature selection and model evaluation. The base model with default features were having better classification and less overfitting as compared to other models. Hence, hyperparameter tuning was not needed.
2. app.py - This contains Flask APIs that receive features of mushroom through frontend web app or API calls, computes the precited value based on our model and returns it back to frontend web app. In addition to it, features and predicted class is stored in "history.csv" as logs.
4. templates - This folder contains the HTML template to allow user to enter features of mushroom and display the predicted class of mushroom - "Edible" or "Poisonous". Also, it displays historical data of each mushroom classification based on input features.

### Running the project
1. Ensure that you are in the project home directory.
A serialized version of our model is present as file "mushroom_model.pkl" in the following directory "model" inside home directory.

2. Run app.py using below command to start Flask API
```
python app.py
```
Flask will run on port 12345.

3. Navigate to URL http://localhost:12345

Select the features of mushroom based on user form provided and hit Submit.

If everything goes well, you should be able to see the predicted class of mushroom on the HTML page!
http://localhost:12345/predict
For each prediction, Classification History table is updated with latest predictions on top.

If you want to get the log in a csv format, it's generated in the project home directory as "history.csv".

