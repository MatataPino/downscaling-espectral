# -*- coding: utf-8 -*-
"""Paso 2 -- PCA sobre los espectros estandarizados y seleccion por maxima
disimilitud. Escribe seleccion_mda.npz.
"""
import time
import numpy as np
from _comun import iniciar, C
from downscaling.espectros import abrir_nodo
from downscaling.pca import pca_estandarizado, tabla_error
from downscaling.seleccion import mda

cfg = iniciar(__doc__); t0 = time.time()
B = np.load(C.salida(cfg, 'base_eof_nodo.npz'))
frec = B['frec'].ravel(); dirs = B['dirs'].ravel()
f, g = abrir_nodo(cfg); Spec = g['Spec']
d = int(cfg['pca']['d']); M = int(cfg['seleccion']['mda_casos'])

R = pca_estandarizado(Spec, frec, dirs, int(cfg['nodo']['bloque']), d)
print('[2] celdas con varianza: %d de %d' % (R['val'].sum(), len(R['val'])))
print('    error de reconstruccion en el clima operacional (criterio de Camus et al., 2013)')
print('      %4s %8s | %8s %9s %10s' % ('d', 'var[%]', 'Hs [m]', 'Tm01 [s]', 'th_m [deg]'))
for fila in tabla_error(Spec, frec, dirs, R, cfg['pca']['tabla_d']):
    print('      %4d %8.2f | %8.3f %9.3f %10.2f' % fila)
print('    d adoptado = %d (%.2f %% de varianza)' % (d, R['vexp'][d-1]))

sel, lo, hi = mda(R['C'], R['Hs'], M)
print('[3] maxima disimilitud: %d casos | Hs %.2f-%.2f m | Tp %.1f-%.1f s'
      % (M, R['Hs'][sel].min(), R['Hs'][sel].max(), R['Tp'][sel].min(), R['Tp'][sel].max()))
np.savez_compressed(C.salida(cfg, 'seleccion_mda.npz'),
    sel=sel, C=R['C'][sel], Hs=R['Hs'][sel], Tp=R['Tp'][sel], Dp=R['Dp'][sel],
    mu=R['mu'], sd=R['sd'], val=R['val'], EOF=R['EOF'], d=d, var_exp=R['vexp'][:d],
    lo=lo, hi=hi, M=M)
print('    -> seleccion_mda.npz  (%.1f min)' % ((time.time()-t0)/60))
