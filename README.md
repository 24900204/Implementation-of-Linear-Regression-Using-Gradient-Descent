# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RITHIKA L
RegisterNumber: 212224230231
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv('/content/50_Startups.csv', header=None)
print(data.head())
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Name: RITHIKA L')
print('Register No.:212224230231')
print(X1_Scaled)
theta = linear_regression(X1_Scaled, y1_Scaled)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="686" height="132" alt="image" src="https://github.com/user-attachments/assets/113aca28-8ca9-400d-a6a1-94bbfafb42a6" />


<img width="392" height="842" alt="image" src="https://github.com/user-attachments/assets/bcf77995-d8ed-438e-9941-852559c8c5b2" />


<img width="392" height="237" alt="image" src="https://github.com/user-attachments/assets/9243a256-9579-4e34-a202-d4f0f4b28ce7" />


<img width="287" height="567" alt="image" src="https://github.com/user-attachments/assets/f5ae6acc-d4af-4bf1-8d75-8d9b72452e94" />


<img width="531" height="558" alt="image" src="https://github.com/user-attachments/assets/257dab1a-386b-4a8b-9828-45309d002aea" />


<img width="530" height="552" alt="image" src="https://github.com/user-attachments/assets/a21a3266-67a4-4071-8a80-c920d17a4c03" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
