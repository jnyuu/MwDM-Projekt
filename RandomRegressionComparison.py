import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(0)
X = 2 * np.random.rand(100, 1) 
y = 2 * X + 1 + np.random.randn(100, 1)  
degree = 4


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

# linear regression
result = linear_regression(X,y)
x_values = np.linspace(0, 2, 100) 
y_values = result[0] * x_values + result[1]

# polynomial regression
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", s=30, marker="o", label="Training data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label=f"Polynomial Regression (degree={degree})")
plt.plot(x_values, y_values,color='green', linewidth=2, label=f"Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.xlim(0, 2)
plt.ylim(0, 7)
plt.title("Polynomial Regression")
plt.legend()
plt.show()

