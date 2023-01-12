'''
1. Generate 1000 random points (x,y)
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9)

# Generate Random Points
N = 1000

x = np.random.randn(N)

for i in range(N):
    sigma = np.random.randint(1, 2)
    y = x + sigma * np.random.randn(N)

plt.scatter(x, y)

#%%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                random_state=0)

#%%

'''
2. Build a simple linear regression from scratch

'''

def simple_linear_regression(x, y):
    n = len(x)

    # Sum(x * y)
    xy_sum = 0
    for i in range(n):
        xy_sum += x[i] * y[i]

    # Sum(x^2)
    xSq_sum = 0
    for i in range(n):
        xSq_sum += x[i] * x[i]

    b1 = (xy_sum - n * x.mean() * y.mean()) / (xSq_sum - n * x.mean() * x.mean())
    b0 = y.mean() - b1 * x.mean()

    return b0, b1

m_lr, c_lr = simple_linear_regression(x, y)

#%%

'''
3. Use gradient descent to find the optimal slope and intercept value
'''

def sumSquaredError(y, yPred):
    sse = 0
    
    for i in range(len(y)):
        e = (y[i] - yPred[i]) ** 2
        sse += e

    return sse

def m_deriv(x, y, yPred):
    error = (x * (y - yPred)).sum()
  
    return -2 * error

def c_deriv(x, y, yPred):
    error = (y - yPred).sum()
  
    return -2 * error


# Learning Rate
alpha = 0.00001

m = np.random.random()
c = np.random.random()

yPred = m * x_train + c
cost = sumSquaredError(y_train, yPred)

for i in range(1000):
    m = m - alpha * m_deriv(x_train, y_train, yPred)
    c = c - alpha * c_deriv(x_train, y_train, yPred)
    
    yPred = (m * x_train) + c

    cost = sumSquaredError(y_train, yPred)

#%%

'''
4. Apply sklearn on the dataset and find the best fit slope and intercept. 
Compare the values obtained in step 3 and 4 and write your inference from that.
'''

    
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# x_train & y_train must be 2-Dimensional, so use 'reshape()'
reg.fit(x_train.reshape(-1, 1), y_train)

yPred_sklearn = reg.predict(x_test.reshape(-1, 1))
    
#%%
   
# Output

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

r2 = r2_score(y_test,yPred_sklearn)
print("r2 =", r2)
 
print("\n*** Linear Regression Model ***")
print(f"Slope: {c_lr}")
print(f"Intercept: {m_lr}")

print("\n*** Without scikit-learn ***")
print(f"Slope: {m}")
print(f"Intercept: {c}")

print("\n*** With scikit-learn ***")
print(f"Slope: {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

print("\nInference: ")
print("Gradient descent [GD] model depends on Learning Rate.")
print("GD is faster than calculating all error values for 'm' and 'c'. ")


#%%


