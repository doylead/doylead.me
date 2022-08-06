## Necessary imports
import numpy as np
from sklearn import linear_model
from tabulate import tabulate
import pylab as plt

## For repeatable results
np.random.seed(42)

## Construct the data
gen_m = np.random.rand()
gen_b = np.random.rand()
x = np.arange(0,21).reshape(-1, 1)

# Includes some random noise around each point
y = gen_m * x + gen_b + np.random.rand(*x.shape)

## Fit a linear model using sklearn
reg = linear_model.LinearRegression()
reg.fit(x, y)

# We extract scalar values from the return format
# scikit-learn provides
skl_m = reg.coef_[0][0]
skl_b = reg.intercept_[0]

## Fit a model using our techniques (specifically
## using Equation (6) and abstract approaches)
## We need the transpose to keep inner
## dimensions consistent
x_mean = np.average(x)
y_mean = np.average(y)
x_T = np.transpose(x)

# The output of both dot products is a 1x1 matrix.
# We use array[0][0] to extract the first/only
# value of this matrix as a scalar
afit_m = np.dot(x_T-x_mean, y-y_mean)[0][0] / \
    np.dot(x_T-x_mean, x-x_mean)[0][0]
afit_b = y_mean - afit_m*x_mean
afit_y = afit_m * x + afit_b

## Visualize results
# Compare parameters from scikit-learn and
# our methods
table_data = [
    ["scikit-learn", skl_m, skl_b],
    ["analytical", afit_m, afit_b]
]
print(tabulate(table_data, 
    headers=["Method", "Slope", "Intercept"]))

# Make a plot for the data
plt.plot(x, y, 
    ms=15,
    lw=0, 
    marker='o', 
    color='blue',
    label="Sample Data")

# Show our fit
plt.plot(x, afit_y,
    ms=0,
    lw=2,
    ls='--',
    color='black',
    label="Linear Fit")

# Tidy things up and show the plot
plt.tight_layout()
plt.legend(frameon=False, prop={'size': 20})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()