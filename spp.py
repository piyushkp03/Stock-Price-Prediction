import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import confusion_matrix, accuracy_score 


sdset=pd.read_csv("Datasets/CIPLA.csv")
print(sdset.shape)
print(sdset.describe())
print(sdset.head())
#plt.plot(sdset['Open'])
#plt.show()

x=sdset[['Open','Low', 'High', 'Volume']]
y=sdset['Close']
x1,x2,y1,y2=train_test_split(x,y,random_state=0)
#print(x1.shape,x2.shape)

regr=LinearRegression()
regr.fit(x1,y1)
print(regr,regr.coef_, regr.intercept_)

pred=regr.predict(x2)

xx=pd.DataFrame({"Open":100,"Low":100,"High":200,"Volume":230},index=[0])
pred1=regr.predict(xx)
print("The predicted value s",pred1)
#print(y2[:15])
#print(pred[:15])


data = {'Actual':y2,
        'Predicted':pred}

df=pd.DataFrame(data)

print(df,xx)
print(regr.score(x2,y2))
dfr=df.head(30)
dfr.plot(kind="bar")
plt.show()