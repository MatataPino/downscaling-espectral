# -*- coding: utf-8 -*-
"""Paso 6 -- Interpolacion RBF y reconstruccion del registro completo en el
punto de destino. Escribe reconstruccion_<punto>.npz.
"""
import numpy as np
from _comun import iniciar, C
from downscaling.reconstruccion import reconstruir

cfg = iniciar(__doc__)
R = dict(np.load(C.salida(cfg, 'seleccion_mda.npz')))
B = dict(np.load(C.salida(cfg, 'base_eof_nodo.npz')))
k = np.load(C.salida(cfg, 'seleccion_kmedias.npz'))['sel']
ver = np.load(C.salida(cfg, 'verificacion.npy'))
print('[6] interpolacion y reconstruccion')
res, v = reconstruir(cfg, R['sel'], k, ver, R, B)
if v:
    print('    verificacion (%d estados ajenos al entrenamiento)' % v['n'])
    print('      %-6s %9s %9s %9s %9s' % ('', 'sesgo', 'RMSE', 'med|e|', 'r'))
    for n in ('Hs', 'Tp', 'Dpeak'):
        print('      %-6s %+9.3f %9.3f %9.3f %9.3f' % ((n,)+v[n]))
out = C.salida(cfg, 'reconstruccion_%s.npz' % cfg['rbf']['punto'])
np.savez_compressed(out, **res)
print('    -> %s' % out)
