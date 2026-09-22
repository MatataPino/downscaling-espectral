# -*- coding: utf-8 -*-
"""Lectura del archivo de configuracion del sitio.

Toda la informacion especifica de un sitio vive en un archivo TOML de la carpeta
config/: ubicacion de los datos, coordenadas del nodo, carpetas y puntos de
salida de SWAN y parametros de la metodologia. Los programas no contienen ningun valor
propio del sitio.

Las rutas admiten variables de entorno (${DATOS}, %DATOS%). Las relativas se
resuelven asi:
  - rutas.datos y rutas.salida, respecto de la raiz del repositorio;
  - el resto de rutas de datos, respecto de rutas.datos.
"""
import os
import tomllib
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent


def cargar(ruta):
    ruta = Path(ruta).resolve()
    with open(ruta, 'rb') as f:
        cfg = tomllib.load(f)
    cfg['_archivo'] = str(ruta)
    return cfg


def _expandir(p):
    q = os.path.expanduser(os.path.expandvars(str(p)))
    if '$' in q or ('%' in q and q.count('%') >= 2):
        raise ValueError(
            "La ruta '%s' contiene una variable de entorno sin definir. "
            "Definala o escriba la ruta completa en el archivo de configuracion." % p)
    return q


def _base(cfg, clave):
    b = Path(_expandir(cfg['rutas'][clave]))
    return b if b.is_absolute() else RAIZ_REPO / b


def datos(cfg, rel):
    """Ruta de un dato de entrada, relativa a rutas.datos."""
    p = Path(_expandir(rel))
    return p if p.is_absolute() else _base(cfg, 'datos') / p


def salida(cfg, nombre=None):
    """Carpeta de resultados intermedios del pipeline; se crea si no existe."""
    d = _base(cfg, 'salida')
    d.mkdir(parents=True, exist_ok=True)
    return d if nombre is None else d / nombre


def raiz_swan(cfg):
    """Directorio de trabajo de SWAN: contiene las carpetas de casos."""
    return datos(cfg, cfg['swan']['raiz'])


def carpeta_casos(cfg, conjunto):
    """Carpeta de un conjunto de casos, relativa a la raiz de SWAN."""
    return cfg['swan']['carpetas'][conjunto]


def punto(cfg, nombre):
    for p in cfg['swan']['puntos']:
        if p['nombre'] == nombre:
            return p
    raise KeyError("El punto '%s' no esta definido en [[swan.puntos]]" % nombre)
