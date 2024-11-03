import numpy as np
import matplotlib.pyplot as plt
import unittest

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

class TestLinearRegression(unittest.TestCase):
    
    def test_linear_regression_output_type(self):
        # Test to check if the output is a list
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [3], [4]])
        result = linear_regression(X, y)
        self.assertIsInstance(result, list, "Output should be a list")
    
    def test_linear_regression_slope_intercept(self):
        # Test to check if slope and intercept are correct for simple data
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [3], [4]])
        result = linear_regression(X, y)
        
        # For y = X + 1, slope should be close to 1 and intercept to 1
        self.assertAlmostEqual(result[0], 1, places=2, msg="Slope should be close to 1")
        self.assertAlmostEqual(result[1], 1, places=2, msg="Intercept should be close to 1")

    def test_linear_regression_no_variation(self):
        # Test with no variation in y values
        X = np.array([[1], [2], [3]])
        y = np.array([[5], [5], [5]])
        result = linear_regression(X, y)
        
        # Slope should be 0 and intercept should be equal to y value
        self.assertAlmostEqual(result[0], 0, places=2, msg="Slope should be 0")
        self.assertAlmostEqual(result[1], 5, places=2, msg="Intercept should be equal to constant y value")

    def test_linear_regression_large_data(self):
        # Test with larger dataset with known slope and intercept
        np.random.seed(0)
        X = 2 * np.random.rand(100, 1)
        y = 3 * X + 2 + np.random.randn(100, 1) 
        result = linear_regression(X, y)
        
        # Slope should be close to 3 and intercept close to 2
        self.assertAlmostEqual(result[0], 3, delta=0.5, msg="Slope should be close to 3")
        self.assertAlmostEqual(result[1], 2, delta=0.5, msg="Intercept should be close to 2")

if __name__ == '__main__':
    unittest.main()