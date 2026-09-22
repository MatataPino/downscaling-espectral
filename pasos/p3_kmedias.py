# -*- coding: utf-8 -*-
"""Paso 3 -- Seleccion por K-medias sobre el clima operacional y estados de
verificacion. Escribe seleccion_kmedias.npz y verificacion.npy.
"""
import numpy as np
from _comun import iniciar, C
from downscaling.espectros import abrir_nodo
from downscaling.seleccion import kmedias, verificacion

cfg = iniciar(__doc__); s = cfg['seleccion']
B = dict(np.load(C.salida(cfg, 'base_eof_nodo.npz')))
f, g = abrir_nodo(cfg)
print('[3] K-medias')
res, Hs_nodo = kmedias(g['Spec'], B, int(cfg['nodo']['bloque']), int(s['kmedias_componentes']),
                       float(s['kmedias_hs_max']), int(s['kmedias_casos']), s['kmedias_k_codo'],
                       int(s['kmedias_muestra']), int(s['semilla']))
print('    %d medoides | Hs %.2f-%.2f m' % (len(res['sel']), res['Hs'].min(), res['Hs'].max()))
np.savez(C.salida(cfg, 'seleccion_kmedias.npz'), **res)

if s.get('verificacion_indices'):
    ruta = C.RAIZ_REPO / s['verificacion_indices']
    ver = np.loadtxt(ruta, dtype=np.int64, comments='#')
    print('[4] verificacion: %d estados fijados en %s' % (len(ver), ruta.name))
else:
    sel_mda = np.load(C.salida(cfg, 'seleccion_mda.npz'))['sel']
    ver = verificacion(Hs_nodo, float(s['kmedias_hs_max']), int(s['verificacion_casos']),
                       np.r_[sel_mda, res['sel']], int(s['semilla']))
    print('[4] verificacion: %d estados sorteados (semilla %d)' % (len(ver), int(s['semilla'])))
np.save(C.salida(cfg, 'verificacion.npy'), ver)
print('    -> seleccion_kmedias.npz, verificacion.npy')
