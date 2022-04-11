import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
import joblib

#feed the Decision tree with relevant datasets to trigger the correct response
df = pd.read_csv(r"trafficdata.csv")
#variables involved
x = df.loc[:, ["x", "y", "Congestion","Satisfaction","Accident"]]
#experimental result outcome
y = df.loc[:, ["default"]]

#Model One: Decision Tree
model = tree.DecisionTreeClassifier(max_depth=2)
model.fit(x,y)
pred = model.predict(x)
cm = confusion_matrix(y, pred)
print(cm)
joblib.dump(model, "CART")

#Model Two:Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(ccp_alpha=0.0384)
model.fit(x,y)
pred = model.predict(x)
cm = confusion_matrix(y,pred)
print(cm)
joblib.dump(model, "RF")

#3rd model-GradientBoosting

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(min_samples_split=30, random_state=260322)
model.fit(x,y)
pred = model.predict(x)
cm = confusion_matrix(y,pred)
print(cm)
joblib.dump(model,"GB")

#Insert optimisation line if needed