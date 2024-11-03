import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = 2 * np.random.rand(100, 1) 
y = 2 * X + 1 + np.random.randn(100, 1)  

def linear_regression(X, y):

    AVG_X = sum(X) / len(X)
    AVG_Y = sum(y) / len(y)

    # print(AVG_X)
    # print(AVG_Y)

    SUM_X = AVG_X - X

    # print(SUM_X)

    SUM_XX = sum(SUM_X ** 2)

    # print(SUM_XX)

    SUM_Y = AVG_Y - y

    # print(SUM_Y)

    SUM_XY = sum(SUM_X * SUM_Y)

    # print(SUM_XY)

    slope = SUM_XY / SUM_XX
    intercept =  AVG_Y - slope * AVG_X 

    print("Slope:", slope)
    print("Intercept:", intercept)
    return([slope,intercept])

# apply linear regression
result = linear_regression(X,y)
x_values = np.linspace(0, 2, 100) 
y_values = result[0] * x_values + result[1]

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values,color='red')
plt.scatter(X, y, color='blue', label='Data points')
plt.title('Linear Regression Example from Scratch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# https://www.voxco.com/blog/how-can-you-calculate-linear-regression/
