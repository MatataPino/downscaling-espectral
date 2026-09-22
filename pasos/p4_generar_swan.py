# -*- coding: utf-8 -*-
"""Paso 4 -- Espectros de contorno y archivos de entrada de SWAN para los tres
conjuntos: maxima disimilitud, K-medias y verificacion.
"""
import numpy as np
from _comun import iniciar, C
from downscaling.espectros import abrir_nodo
from downscaling.swan import escribir_casos

cfg = iniciar(__doc__)
B = np.load(C.salida(cfg, 'base_eof_nodo.npz')); frec = B['frec'].ravel(); dirs = B['dirs'].ravel()
f, g = abrir_nodo(cfg); Spec = g['Spec']
m = np.load(C.salida(cfg, 'seleccion_mda.npz')); k = np.load(C.salida(cfg, 'seleccion_kmedias.npz'))
ver = np.load(C.salida(cfg, 'verificacion.npy'))
print('[4] casos de SWAN')
escribir_casos(cfg, 'mda', m['sel'], Spec, frec, dirs, np.c_[m['Hs'], m['Tp'], m['Dp']])
escribir_casos(cfg, 'kmedias', k['sel'], Spec, frec, dirs, np.c_[k['Hs'], k['Tp'], k['Dp']])
escribir_casos(cfg, 'verificacion', ver, Spec, frec, dirs)
