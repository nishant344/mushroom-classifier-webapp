import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load("model/mushroom_model.pkl")

@app.route('/')
def home():
    return render_template('main.html')

#@app.route('/predict',methods=['POST'])
#def predict():
#    '''
#    For rendering results on HTML GUI
#    '''
#    features= [x for x in request.form.values()]
#    print(features)

@app.route('/predict',methods=['POST'])
def predict():
    if GB:
        try:
            int_features = [str(x) for x in request.form.values()]
            #final_features = [np.array(int_features)]
            top_features_include = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type']
#            json_ = request.form.values()
            dict_use = dict(zip(top_features_include, int_features))
            print('Predicting for paramters',int_features,top_features_include)
            
            query = pd.get_dummies(pd.DataFrame(dict_use,index=[0]))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(GB.predict(query))
            print ('Result: ',prediction)

            return jsonify({'prediction': str(prediction)})
            return render_template('main.html', features='Mushroom is'.format(prediction))

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
#    int_features = [int(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    prediction = model.predict(final_features)

#    output = round(prediction[0], 2)

#    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

GB = joblib.load("model/mushroom_model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model/model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')
print(model_columns)

app.run(port=port, debug=True)