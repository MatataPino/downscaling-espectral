# -*- coding: utf-8 -*-
"""Paso 1 -- Base EOF centrada del registro del nodo oceanico.

La emplea la seleccion por K-medias (paso 3). Escribe base_eof_nodo.npz.
"""
import numpy as np
from _comun import iniciar, C
from downscaling.espectros import abrir_nodo
from downscaling.pca import base_centrada

cfg = iniciar(__doc__)
f, g = abrir_nodo(cfg)
frec = np.array(g['frec']).ravel(); dirs = np.array(g['dir']).ravel()
print('[1] %d espectros de %d x %d celdas' % (g['Spec'].shape[0], len(dirs), len(frec)))
res = base_centrada(g['Spec'], frec, dirs, int(cfg['nodo']['bloque']), int(cfg['pca']['modos_base']))
cum = np.cumsum(res['var_ratio'])
print('    PC1 %.1f %% | 9 modos %.1f %% | %d modos guardados'
      % (100*res['var_ratio'][0], 100*cum[8], len(res['var_ratio'])))
np.savez(C.salida(cfg, 'base_eof_nodo.npz'), **res)
print('    -> %s' % C.salida(cfg, 'base_eof_nodo.npz'))
