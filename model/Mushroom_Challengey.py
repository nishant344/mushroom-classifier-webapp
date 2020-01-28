""" Mushroom CHallenge """
""" Main model """

# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

# Load data from csv
data = pd.read_csv('mushrooms.csv')

# Top features from model.feature_importance done during EDA
top_features_include = ['odor', 'bruises', 'gill-spacing', 'gill-size', 'ring-type']
X = data[top_features_include]

# Labelling current data values with relevant text values
odor={"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}
X["odor"]=X["odor"].replace(odor)
bruises={"t":"yes","f":"no"}
X["bruises"]=X["bruises"].replace(bruises)
gill_spacing={"c":"close","w":"crowded"}
X["gill-spacing"]=X["gill-spacing"].replace(gill_spacing)
gill_size={"b":"broad","n":"narrow"}
X["gill-size"]=X["gill-size"].replace(gill_size)
ring_type={"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}
X["ring-type"]=X["ring-type"].replace(ring_type)

Y=data['class'] #Label

# One Hot Encoding categorical variables
Y = pd.get_dummies(Y,prefix_sep='_', drop_first=True)
X = pd.get_dummies(X)

n=42 # Random State

# Split dataset into training and test data
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3, random_state=n)

# Initialise best classifier found during EDA (less overfitting) and fit the classifier
GB = GradientBoostingClassifier(random_state=n)
classifier = GB.fit(X_train,Y_train.values.ravel())

# See if the model is reasonable.
print("Score: ", classifier.score(X_test, Y_test))

# Get prediction results are relevant metrics
y_pred = classifier.predict(X_test)
class_eval = classification_report(Y_test, y_pred)
cm = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)

# Pickle to save the model for use in our API.
from sklearn.externals import joblib
joblib.dump(GB, 'mushroom_model.pkl')
print("Model dumped!")

# Load the model that you just saved
GB = joblib.load('mushroom_model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
