# -*- coding: utf-8 -*-
"""Lectura de espectros: el registro del nodo oceanico y las salidas de SWAN.

Nodo oceanico: archivo HDF5 (.mat v7.3) con un grupo que contiene
  Spec  (Nt, Ndir, Nf)  densidad espectral en m2/Hz/rad
  frec  (Nf,)           frecuencias en Hz
  dir   (Ndir,)         direcciones nauticas en grados
  time  (4, Nt)         anio, mes, dia, hora
"""
import numpy as np
import h5py
from . import config as C


def abrir_nodo(cfg):
    """Devuelve (archivo, grupo) del registro del nodo oceanico."""
    f = h5py.File(C.datos(cfg, cfg['nodo']['archivo']), 'r')
    return f, f[cfg['nodo']['grupo']]


def nombre_salida(punto, k):
    """Nombre, sin extension, con que se esperan las salidas de SWAN del caso k."""
    return "%s_%03d" % (punto.lower(), k)


def leer_spc(p):
    """Espectro 2D de un fichero .spc de SWAN -> (frec, dir, espectros)."""
    L = open(p, encoding='latin-1').read().splitlines(); i = 0
    while 'FREQ' not in L[i]: i += 1
    n1 = int(L[i+1].split()[0]); fq = np.array([float(L[i+2+j]) for j in range(n1)]); i += 2+n1
    while 'DIR' not in L[i]: i += 1
    n2 = int(L[i+1].split()[0]); dd = np.array([float(L[i+2+j]) for j in range(n2)]) % 360.; i += 2+n2
    while 'QUANT' not in L[i]: i += 1
    nq = int(L[i+1].split()[0]); i = i+2+3*nq; sp = []
    while i < len(L):
        k = L[i].strip()
        if k.startswith('FACTOR'):
            f_ = float(L[i+1]); i += 2
            sp.append(np.array([[float(x) for x in L[i+r].split()] for r in range(n1)])*f_); i += n1
        elif k.startswith('ZERO'):
            sp.append(np.zeros((n1, n2))); i += 1
        else:
            i += 1
    return fq, dd, np.array(sp)


def leer_tab(p):
    """Primera fila de un .tab de SWAN -> (Hsig, TPsmoo)."""
    for ln in open(p, encoding='latin-1'):
        if ln.startswith('%') or not ln.strip(): continue
        v = ln.split(); return float(v[0]), float(v[2])


def tab_con_datos(p):
    try:
        return any(l.strip() and not l.startswith('%')
                   for l in open(p, encoding='latin-1', errors='replace'))
    except OSError:
        return False


def dir_pico(fq, dd, S):
    """Direccion de pico con refinamiento circular sobre los tres sectores vecinos."""
    dfq = np.abs(np.gradient(fq)); Sd = (S*dfq[:, None]).sum(0); n = len(dd); i0 = int(Sd.argmax())
    ix = [(i0+k) % n for k in (-1, 0, 1)]; w = Sd[ix]; th = np.deg2rad(dd[ix])
    return np.rad2deg(np.arctan2((w*np.sin(th)).sum(), (w*np.cos(th)).sum())) % 360
