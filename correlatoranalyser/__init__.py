# Class that stores/organizes results of fits
# result of fit(...)
from .fitResult import FitResult

# Collection of FitResults. It allows to simplify model averaging etc.
from .fitState import FitState

# A method that with provided data performs fit and return a FitResult
from .fit import fit

# A explicit linear regression
from .linearRegression import linear_regression
