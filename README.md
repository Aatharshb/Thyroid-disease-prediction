# Thyroid-disease-prediction
import pandas as pd
df = pd.read_csv("thyroid.csv") 
import numpy as np 
df=df.replace({"?":np.NAN})
df=df.replace({"P":0,"N":1})
df=df.replace({"M":1,"F":2})
df=df.replace({"t":1,"f":0})
dele=["TBG","referral source",'TSH measured','T3 measured','TT4 measured','T4U measured','FTI measured',"TBG measured"]
for i in dele:
del df[i] 
df.columns
Index(['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', ‘FTI’, binaryClass]) 
from sklearn.impute import SimpleImputera = SimpleImputer(strategy='mean') 
df['TSH'] = a.fit_transform(df[['TSH']]) 
df['sex'] = a.fit_transform(df[['sex']]) 
df['TT4'] = a.fit_transform(df[['TT4']]) 
df['FTI'] = a.fit_transform(df[['FTI']]) 
df['T4U'] = a.fit_transform(df[['T4U']]) 
df['age'] = a.fit_transform(df[['age']]) 
df['T3'] = a.fit_transform(df[['T3']]) 
from sklearn.model_selection import train_test_split as tts 
y= df["binaryClass"] 
x= df[['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']] 
x_train, x_test, y_train, y_test = tts(x, y, train_size=0.6, random_state=0) 
from sklearn.tree import DecisionTreeClassifier 
dtcmodel = DecisionTreeClassifier() 
dtc = dtcmodel.fit(x_train,y_train) 
dtcy_pred = dtcmodel.predict(x_test) 
from sklearn.metrics import accuracy_score as acs 
print("DecisionTreeClassifier Accuracy :",acs(dtcy_pred,y_test)) 
Decisiontreeclassifier Accuracy : 0.9973492379058979 
from sklearn.neighbors import KNeighborsClassifier 
knmodel = KNeighborsClassifier(n_neighbors=3) 
knn = knmodel.fit(x_train,y_train) 
knny_pred = knmodel.predict(x_test) 
print("KNeighborsClassifier Accuracy :",acs(knny_pred,y_test)) 
KNeighborsClassifier Accuracy : 0.950960901259112 
from sklearn.svm import SVC 
svcmodel = SVC() 
svc = svcmodel.fit(x_train,y_train) 
svcy_pred = svcmodel.predict(x_test) 
print("SupportVectorMachine Accuracy :",acs(svcy_pred,y_test)) 
SupportVectorMachine Accuracy : 0.9443339960238568 
import numpy as np 
import matplotlib.pyplot as plt 
data = {'KNN':0.950960901259112, 'DTC':0.9973492379058979, 'SVM':0.9443339960238568} 
alg = list(data.keys()) 
val = list(data.values()) 
fig = plt.figure(figsize = (10, 5)) 
plt.bar(alg, val, color ='b') 
plt.xlabel("Algorithms") plt.ylabel("Accuracy") plt.title("Accuracy Level”) 
plt.show() 
import matplotlib.pyplot as plt 
from sklearn import metrics 
confmatrix = metrics.confusion_matrix(y_test, dtcy_pred) 
cmdisp = metrics.ConfusionMatrixDisplay(confusion_matrix = confmatrix, display_label 
s = [False, True]) 
cmdisp.plot() 
plt.show() 
l1=[] 
prediction = dtcmodel.predict(x_test) 
l1.append(prediction) 
for i in l1: 
for j in i: 
print(j) 
x = input().split(",") 
l=[] 
for i in x: 
l.append(float(i)) 
pred = dtcmodel.predict([l]) . 
dic ={1:"Male",2:"Female"} . 
perg = {1:"Pregnant", 2:"Not pregnant"} 
if l[1] == 1: 
print(dic[l[1]]) 
else : 
print(dic[l[1]]) 
if l[6] ==1: 
print(perg[1]) 
else: 
print(perg[2]) 
if pred == 1: 
print("Postive") 
else: 
print("Negative")
