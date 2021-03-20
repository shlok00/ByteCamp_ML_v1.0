from sklearn import datasets
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics


data = pd.read_json("test2.json") #API url here
#data -- review

data = pd.get_dummies(data=data, columns=['City', 'State', 'Department', 'Gender'])

y = np.array(data['Viewed'])
data= data.drop(['Viewed','Date','Name','MaritalStatus'], axis = 1)
feature_list = list(data.columns)
X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
rf=RandomForestClassifier(n_estimators=30)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print("Accuracy:" +str(metrics.accuracy_score(y_test, y_pred)*100) +"%")
