# -*- coding: utf-8 -*-
"""Downscaling espectral hibrido de oleaje.

Metodologia de Camus et al. (2011a, 2011b, 2013) con entrada y salida
espectrales: reduccion por PCA, seleccion de los estados de mar a propagar por maxima
disimilitud y K-medias, e interpolacion por funciones de base radial sobre
los espectros que SWAN propago para esos estados.
"""
