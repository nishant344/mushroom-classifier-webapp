import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.externals import joblib
import os
import traceback

app = Flask(__name__)
model = joblib.load("model/mushroom_model.pkl")

@app.route('/')
def home():
    headers = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type','mushroom_class']
    df_history=pd.DataFrame(columns=headers)
    if not os.path.isfile(r'history.csv'): # create an empty history with header if it is not present
        df_history.to_csv (r'history.csv', index=None, header=True)
    else: # else load existing history
        df_history=pd.read_csv(r'history.csv')
    return render_template('main.html', logs=df_history.values)
    

@app.route('/predict',methods=['POST'])
def predict():
    if GB:
        try:
            # Build dataframe from user input
            input_vals = [str(x) for x in request.form.values()]
            features = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type']
            feature_vals = dict(zip(features, input_vals))
            df_feature_vals = pd.get_dummies(pd.DataFrame(feature_vals,index=[0]))
            df_feature_vals = df_feature_vals.reindex(columns=model_columns, fill_value=0)

            # Predict mushroom class
            prediction = GB.predict(df_feature_vals)
            if prediction == 0:
                mushroom_class = "Edible"
            else:
                mushroom_class = "Poisonous"
            pred = {'mushroom_class': mushroom_class}

            # Store result in history
            history_dict = {**feature_vals, **pred}
            df_pred = pd.DataFrame(history_dict,index=[0])
            df_history = pd.read_csv(r'history.csv')
            df_pred = pd.concat([df_pred, df_history]).reset_index(drop=True)
            df_pred.to_csv (r'history.csv', index=None, header=True)

            # Render html with updated result and history
            output = 'Your Mushroom is : '+ mushroom_class
            return render_template('main.html', features=output, logs=df_pred.values)
        except:
            print("Error encountered during prediction")
    else:
        print('No model present. Please train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

GB = joblib.load("model/mushroom_model.pkl") # Load "model.pkl"
print('Model loaded')
model_columns = joblib.load("model/model_columns.pkl") # Load "model_columns.pkl"
print('Model columns loaded')
print('Starting app at http://localhost:12345')
app.run(port=port, debug=True)