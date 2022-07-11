
import numpy as np
import torch
def func_tdis(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0
    d=d
    # p = np.power(
    #     gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14),
    #     pow)
    # v = 1 / np.pi
    p = np.power( (1 / np.pi) * (v / (d + v * v)),3)
    return np.power(2, np.sum(p))

def Similarity(dist, rho, sigma_array, gamma, v=100):

    if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
    else:
        dist_rho = dist
    dist=dist
    dist_rho[dist_rho < 0] = 0
    # Pij = torch.pow(
    #     gamma * torch.pow(
    #         (1 + dist_rho / v),
    #         -1 * (v + 1) / 2
    #         ) * torch.sqrt(torch.tensor(2 * 3.14)),
    #         2
    #     )
    #P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    Pij = (1 / np.pi) * (v / (dist_rho + v * v))
    # Pij = mylambda * Pij 
    P = torch.pow(Pij,3)

    return P