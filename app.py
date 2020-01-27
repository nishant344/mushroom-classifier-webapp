import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
import pandas as pd
from sklearn.externals import joblib
import os
import traceback

app = Flask(__name__)
model = joblib.load("model/mushroom_model.pkl")

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    if GB:
        try:
            int_features = [str(x) for x in request.form.values()]
            top_features_include = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type']
            
            dict_use = dict(zip(top_features_include, int_features))
            
            query = pd.get_dummies(pd.DataFrame(dict_use,index=[0]))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = GB.predict(query)
            if prediction == 0:
                class_mushroom = "Edible"
            else:
                class_mushroom = "Poisonous"
                
            output = 'Your Mushroom is : '+ class_mushroom
            pred={'mushroom_class':class_mushroom}
            history_dict={**dict_use,**pred}
            
            df_pred=pd.DataFrame(history_dict,index=[0])
            
            # if file does not exist write header
            if not os.path.isfile(r'history.csv'):
               df_pred.to_csv (r'history.csv', index = None, header=True)
            else: # else it will load dataframe from history.csv and append data with current predictions and save it to csv again.
               df1=pd.read_csv(r'history.csv')
               df_pred = pd.concat([df_pred, df1]).reset_index(drop = True)
               df_pred.to_csv (r'history.csv', index = None, header=True)

            return render_template('main.html', features=output)

        except:
            print("error encountered")
#            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

GB = joblib.load("model/mushroom_model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model/model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

app.run(port=port, debug=True)