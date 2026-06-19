import numpy as np 


class Constant:
    def __init__(self, param_key:str = 'c'):
        self.param_key = param_key

    def __call__(self, x, p):
        r"""
            x: np.ndarray, 
                x-axis for the evaluation. Defines a shape of the output array 
            p: dict,
                parameter, uses 'c' as key for the constant 
            
        """
        return np.full_like(x, p[self.param_key])
    
    def grad(self, x, p):
        return np.ones_like(x)
    
    def hessian(self, x, p):
        return np.zeros( (1,1, *x.shape) )
