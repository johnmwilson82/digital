import numpy as np
class GF2_dtype(np.int8):
    def __init__
    def __add__(self, other):

class GF2n_poly(object):
    def __init__ GF2n_poly(degree, init=None):
        self.poly = init or np.zeros(degree)
        self.degree = degree

    def __add__(self, other):
        if other.degree < self.degree:
            self.poly += [0, ] * (other.degree - self.degree)
            self.degree = other.degree
        
        
    @property
    def degree(self):
        return degree
