""" Mushroom CHallenge """
""" Main model """

""" Importing Libraries"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# import pickle

# Data is loaded
data = pd.read_csv('mushrooms.csv')

""" Top features from model.feature_importance done during EDA """
top_features_include = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type']
X = data[top_features_include]
Y=data['class'] #Label

Y = pd.get_dummies(Y,prefix_sep='_', drop_first=True)
X = pd.get_dummies(X)

""" Now we encode all the categories into 1s and 0s by creating a wide dataframe.
Also, one category is dropped in each predictor to solve dummy-variable trap. """
#X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
#y = pd.get_dummies(y, prefix_sep='_', drop_first=True)

n=42

X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3, random_state=n)

GB = GradientBoostingClassifier(random_state=n)
classifier = GB.fit(X_train,Y_train.values.ravel())

# See if the model is reasonable.
print("Score: ", classifier.score(X_test, Y_test))

y_pred = classifier.predict(X_test)
class_eval = classification_report(Y_test, y_pred)
cm = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)

# Pickle to save the model for use in our API.
# pickle.dump(classifier, open("./mushroom_model.pkl", "wb"))
from sklearn.externals import joblib
joblib.dump(GB, 'mushroom_model.pkl')
print("Model dumped!")

# Load the model that you just saved
GB = joblib.load('mushroom_model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
