# -*- coding: utf-8 -*-
"""Paso 5 -- Aplica los interpoladores ya entrenados a un registro nuevo del
mismo nodo: una ampliacion de la serie, por ejemplo.

No vuelve a correr SWAN ni rehace la reduccion: reutiliza la que dejaron los
pasos 1 a 3 y las propagaciones existentes, y evalua sobre el registro que
indica la seccion [aplicar] de la configuracion. El registro nuevo debe ser
del MISMO nodo y tener la misma grilla espectral que el del entrenamiento.

Escribe reconstruccion_<punto>_<etiqueta>.npz.
"""
import numpy as np
from _comun import iniciar, C
from downscaling.reconstruccion import reconstruir

cfg = iniciar(__doc__)
if 'aplicar' not in cfg:
    raise SystemExit('Falta la seccion [aplicar] en %s: archivo, grupo y etiqueta '
                     'del registro nuevo.' % cfg['_archivo'])
ap = cfg['aplicar']
R = dict(np.load(C.salida(cfg, 'seleccion_mda.npz')))
B = dict(np.load(C.salida(cfg, 'base_eof_nodo.npz')))
k = np.load(C.salida(cfg, 'seleccion_kmedias.npz'))['sel']
print('[5] aplicacion a un registro nuevo')
res, _ = reconstruir(cfg, R['sel'], k, None, R, B, registro=ap)
out = C.salida(cfg, 'reconstruccion_%s_%s.npz' % (cfg['rbf']['punto'], ap['etiqueta']))
np.savez_compressed(out, **res)
print('    -> %s' % out)
