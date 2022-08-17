'''This stand-alone script accompanies a blog post available here:


The goal of this script is to generate data which follow a basic bivariate linear relationship and
to show two different methods of calculating the parameters for the best linear fit.  We show that
an approach based on the dot product in an abstract vector space matches trusted results from
scikit-learn
'''

# Necessary imports
import numpy as np
from sklearn import linear_model
from tabulate import tabulate

# For repeatable results
np.random.seed(42)

# Construct the data
gen_t0 = np.random.rand()
gen_t1 = np.random.rand()
gen_t2 = np.random.rand()

# We want to take some care to create X1 and X2 such that they are not correlated
shape = (25,1)
X1 = np.random.rand(*shape)
X2 = np.random.rand(*shape)

# Includes some random noise around each point
Y = gen_t0 + gen_t1 * X1 + gen_t2 * X2 + np.random.normal(size=shape)

# To prepare to fit a linear model using scikit-learn, we'll want to combine X1 and X2 into a
# single matrix
X = np.column_stack((X1, X2))

## Fit a linear model using sklearn
reg = linear_model.LinearRegression()
reg.fit(X, Y)

# We extract scalar values from the return format
# scikit-learn provides
skl_t0 = reg.intercept_[0]
skl_t1 = reg.coef_[0][0]
skl_t2 = reg.coef_[0][1]

# Fit a model using our techniques (specifically using Equations (4), (3), and (2)).  We need
# the transpose to keep inner dimensions consistent
X1_mean = np.average(X1)
X2_mean = np.average(X2)
Y_mean = np.average(Y)

X1_hat = X1 - X1_mean
X2_hat = X2 - X2_mean
Y_hat = Y-Y_mean

X1_hat_T = np.transpose(X1_hat)
X2_hat_T = np.transpose(X2_hat)

# The output of all dot products is a 1x1 matrix.  We use array[0][0] to extract the first/only
# value of this matrix as a scalar
X1_hat_d_X1_hat = np.dot(X1_hat_T, X1_hat)[0][0]
X1_hat_d_X2_hat = np.dot(X1_hat_T, X2_hat)[0][0]
X1_hat_d_Y_hat = np.dot(X1_hat_T, Y_hat)[0][0]
X2_hat_d_X1_hat = X1_hat_d_X2_hat # Symmetry of dot product
X2_hat_d_X2_hat = np.dot(X2_hat_T, X2_hat)[0][0]
X2_hat_d_Y_hat = np.dot(X2_hat_T, Y_hat)[0][0]

# For easier readability we define the numerator and denominator separately
afit_t2_num = X2_hat_d_Y_hat * X1_hat_d_X1_hat - X1_hat_d_Y_hat * X1_hat_d_X2_hat
afit_t2_den = X1_hat_d_X1_hat * X2_hat_d_X2_hat - X1_hat_d_X2_hat**2
afit_t2 = afit_t2_num / afit_t2_den
afit_t1 = X1_hat_d_Y_hat / X1_hat_d_X1_hat - afit_t2 * X1_hat_d_X2_hat / X1_hat_d_X1_hat
afit_t0 = Y_mean - afit_t1 * X1_mean - afit_t2 * X2_mean

# Compare parameters from scikit-learn and our methods
table_data = [
    ["scikit-learn", skl_t0, skl_t1, skl_t2],
    ["analytical", afit_t0, afit_t1, afit_t2]
]

print(tabulate(table_data,
    headers=["Method", "Theta0", "Theta1", "Theta2"]))
