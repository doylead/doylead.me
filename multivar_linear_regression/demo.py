'''This stand-alone script accompanies a blog post available here:
https://doylead.me/?p=232

The goal of this script is to generate data which follow a multivariate linear relationship and
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

n_samples = 200
n_features = 5

# Construct the data
t_shape = (n_features+1,1)
theta_gen = np.random.rand(*t_shape)

# We want to take some care to create our inputs such that they are not correlated
x_shape = (n_samples,n_features)
X = np.random.rand(*x_shape)

# Includes the first column of ones, X_{i,0}=1
X1 = np.column_stack((np.ones(n_samples), X))

# Includes some random noise around each point
Y = np.dot(X1, theta_gen) + np.random.normal(size=(n_samples,1))


## Fit a linear model using sklearn
reg = linear_model.LinearRegression()
reg.fit(X, Y)

# We extract scalar values from the return format
# scikit-learn provides
skl_t0 = reg.intercept_[0]
skl_t1, skl_t2, skl_t3, skl_t4, skl_t5 = reg.coef_[0]

# Fit a model using our techniques (specifically using Equation (3)
X1T = np.transpose(X1)
X1TX1 = np.dot(X1T, X1)
X1TX1_inv = np.linalg.inv(X1TX1)
X1TY = np.dot(X1T, Y)
afit_t0, afit_t1, afit_t2, afit_t3, afit_t4, afit_t5 = np.dot(X1TX1_inv, X1TY)

# Compare parameters from scikit-learn and our methods
table_data = [
    ["scikit-learn", skl_t0, skl_t1, skl_t2, skl_t3, skl_t4, skl_t5],
    ["analytical", afit_t0, afit_t1, afit_t2, afit_t3, afit_t4, afit_t5]
]

print(tabulate(table_data,
    headers=["Method", "Theta0", "Theta1", "Theta2", "Theta3", "Theta4", "Theta5"]))
