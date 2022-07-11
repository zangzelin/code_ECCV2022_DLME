import torch
import numpy as np
from scipy import integrate
import numpy as np

def T1Similarity(dist, rho, sigma_array, gamma, v=100, h=1, pow=2):
    v=100
    # print(h)
    if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
        dist_rho[dist_rho < 0] = 0
    else:
        dist_rho = dist
    Pij = torch.pow( 1.0/(1+dist_rho), pow) * h
    Pij = Pij + Pij.t() - torch.mul(Pij, Pij.t())
    return Pij


def T1SimilarityNumpy(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    
    v=100
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0

    p = np.power( 1.0/(1+d), pow)
    return np.power(2, np.sum(p))

def UMAPSimilarity(dist, rho, sigma_array, gamma, v=100, h=1, pow=2):

    if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
        dist_rho[dist_rho < 0] = 0
    else:
        dist_rho = dist

    dist_rho[dist_rho < 0] = 0
    Pij = torch.pow(
        input=gamma * torch.pow(
            (1 + dist_rho / v),
            exponent= -1 * (v + 1) / 2
            ) * torch.sqrt(torch.tensor(2 * 3.14)),
        exponent=pow
        )

    Pij = Pij + Pij.t() - torch.mul(Pij, Pij.t())
    return Pij


def UMAPSimilarityNumpy(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0
    p = np.power(
        gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14),
        pow
        )
    return np.power(2, np.sum(p))