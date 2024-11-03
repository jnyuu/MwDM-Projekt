import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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



# load kaggle dataset
path = kagglehub.dataset_download("aravinii/house-price-prediction-treated-dataset")
print("Path to dataset files:", path)
data = pd.read_csv(path + "/df_test.csv")  
X = data[['living_in_m2']].values 
y = data[['price']].values 

# apply linear regression
result = linear_regression(X, y)

# generate values for plotting the regression line
x_values = np.linspace(X.min(), X.max(), 100)
y_values = result[0] * x_values + result[1]

# plot the data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(x_values, y_values, color='red', label='Regression line')
plt.title('House Price Prediction with Linear Regression')
plt.xlabel('Living Area (Square Feet)')
plt.ylabel('Sale Price')
plt.legend()
plt.show()


