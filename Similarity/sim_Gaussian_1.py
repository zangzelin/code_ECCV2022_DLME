
import numpy as np
import torch
def func_tdis(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    v=100
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0
    p = np.power(
        gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14),
        pow)
    return np.power(2, np.sum(p))

def Similarity(dist, rho, sigma_array, gamma, v=100, h=1, pow=2):
    # h=v
    v=100
    # print(h,v)

    if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
    else:
        dist_rho = dist
    
    dist_rho[dist_rho < 0] = 0
    Pij = torch.pow( 1.0/(1+dist_rho), 6) * h

    P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    return P