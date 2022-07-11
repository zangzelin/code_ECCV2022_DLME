import torch
import numpy as np

# class EuclideanDistance():

#     def __init__(self,):
#         print('load EuclideanDistance')
        
def EuclideanDistanceNumpy(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    v=100
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0
    p = np.power(
        gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14),
        pow)
    return np.power(2, np.sum(p))
