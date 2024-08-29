# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values.

## Program:
```Python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sajith Ahamed F
RegisterNumber:  212223240144
*/
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
#graph plot for test data
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
#caluculate Mean absolute erroe
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```

## Output:
### head:
![image](https://github.com/user-attachments/assets/8fcf0beb-772a-4949-a3d1-383da7ffe32e)

### tail:
![image](https://github.com/user-attachments/assets/fa70ae56-140e-41b6-9fdf-b66bd304c0a7)

### Output 3:
![image](https://github.com/user-attachments/assets/f03258f1-e678-4df9-b66a-733b14f65670)

### Output 4:
![image](https://github.com/user-attachments/assets/12c31a7a-fa68-40b8-9918-543ae38f0885)

### output 5:
![image](https://github.com/user-attachments/assets/53f10323-baac-4a28-9d43-1d9019c4ad35)

### Output 6:
![image](https://github.com/user-attachments/assets/d25aaa43-7340-43e5-a58c-9305b21c5285)

### Output 7:
![image](https://github.com/user-attachments/assets/fe3a45d8-49cc-4bfe-be29-2b1cc648f102)

### Output 8:
![image](https://github.com/user-attachments/assets/74512aae-b854-48dd-996c-b5e174bbf462)

### Output 9:
![image](https://github.com/user-attachments/assets/423659c4-9da4-4df4-8556-15c96a065914)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
