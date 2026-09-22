# -*- coding: utf-8 -*-
"""Interpolacion por funciones de base radial (Camus et al., 2011b).

    RBF(x) = sum_j a_j exp(-||x - x_j||^2 / 2 sigma^2) + b_0 + sum_l b_l x_l

Nucleo gaussiano, base monomial de grado 1 y ajuste exacto. Los coeficientes
salen de resolver

    [ Phi  P ] [a]   [f]
    [ P^T  0 ] [b] = [0]

y sigma se elige por la validacion cruzada de Rippa (1999), de forma
independiente para cada objetivo.

La matriz del sistema depende solo de sigma y de los puntos, no del objetivo:
se factoriza una vez por sigma y se reutiliza para todos los objetivos. Es la
diferencia de coste frente a BlueMath_tk, que la rehace por objetivo; con 64
objetivos, horas frente a segundos. El resultado es el mismo.
"""
import numpy as np


def malla_sigma(sigma_min, sigma_max, n):
    """Malla logaritmica: el optimo suele caer en la parte baja del rango."""
    return np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), n))


def rippa(Xn, Ys, grid):
    """Sigma optimo por objetivo. Ys: (M, n_objetivos). Devuelve (sigma, error)."""
    M = len(Xn); dpl = Xn.shape[1]
    R2 = ((Xn[:, None, :]-Xn[None, :, :])**2).sum(2); P = np.c_[np.ones(M), Xn]
    Z = np.zeros((dpl+1, dpl+1)); rhs = np.vstack([Ys, np.zeros((dpl+1, Ys.shape[1]))])
    best = np.full(Ys.shape[1], np.inf); sig = np.full(Ys.shape[1], grid[0])
    for c in grid:
        Kk = np.block([[np.exp(-R2/(2*c*c)), P], [P.T, Z]])
        try:
            sol = np.linalg.solve(Kk, rhs); Ki = np.linalg.inv(Kk)
        except np.linalg.LinAlgError:
            continue
        dg = np.diag(Ki)[:M]
        if np.any(np.abs(dg) < 1e-14): continue
        err = np.sqrt(((sol[:M]/dg[:, None])**2).mean(0))     # error de dejar uno fuera
        m = err < best; best[m] = err[m]; sig[m] = c
    return sig, best


def resolver(Xn, c, Y):
    """Coeficientes [a; b] para uno o varios objetivos con el mismo sigma."""
    M = len(Xn); d = Xn.shape[1]
    R2 = ((Xn[:, None, :]-Xn[None, :, :])**2).sum(2); Pm_ = np.c_[np.ones(M), Xn]
    Kk = np.block([[np.exp(-R2/(2*c*c)), Pm_], [Pm_.T, np.zeros((d+1, d+1))]])
    Y = np.atleast_2d(Y.T).T
    return np.linalg.solve(Kk, np.vstack([Y, np.zeros((d+1, Y.shape[1]))]))


def evaluar(R2q, Pq, c, sol, M):
    """Valor del interpolador en los puntos de consulta.
    R2q: distancias al cuadrado a los M centros; Pq: [1, x] de la consulta."""
    return np.exp(-R2q/(2*c*c))@sol[:M]+Pq@sol[M:]
