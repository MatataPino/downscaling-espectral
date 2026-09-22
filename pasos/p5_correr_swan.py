# -*- coding: utf-8 -*-
"""Paso 5 -- Propagacion con SWAN de los tres conjuntos, en paralelo.
Reanudable: si se interrumpe, basta con relanzarlo.
"""
from _comun import iniciar
from downscaling.swan import correr

cfg = iniciar(__doc__)
print('[5] propagacion')
for conjunto in ('mda', 'kmedias', 'verificacion'):
    correr(cfg, conjunto)
