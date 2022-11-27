# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To implement the linear regression using the standard libraries in the python.
2. Use slicing function() for the x,y values.
3. Using sklearn library import training , testing and linear regression modules.
4. Predict the value for the y.
5. Using matplotlib library plot the graphs.
6. Use xlabel for hours and ylabel for scores.
7. End the porgram. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: P Shobika
RegisterNumber:  212221230096

import numpy as np
import pandas as pd
dataset=pd.read_csv('student_scores.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color="purple")
plt.title("h vs s (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,reg.predict(x_test),color="orange")
plt.title("h vs s (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse) 
```

## Output:
![image](https://user-images.githubusercontent.com/94508142/204134420-ae9a2f16-7719-4637-b734-6a423aafd8c3.png)

![image](https://user-images.githubusercontent.com/94508142/204134444-3378988f-cb64-43a9-a6f3-7f5cb6ca80dc.png)

![image](https://user-images.githubusercontent.com/94508142/204134465-aedca5c0-ef37-476a-b90c-28e298f426d9.png)

![image](https://user-images.githubusercontent.com/94508142/204134401-38573624-fbea-4e90-9b38-b863161b1619.png)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
