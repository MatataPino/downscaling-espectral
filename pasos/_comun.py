# -*- coding: utf-8 -*-
"""Arranque comun de los pasos: ruta del paquete y lectura de la configuracion."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from downscaling import config as C   # noqa: E402


def iniciar(descripcion):
    ap = argparse.ArgumentParser(description=descripcion)
    ap.add_argument('--config', default='config/ejemplo.toml',
                    help='archivo de configuracion del sitio')
    a = ap.parse_args()
    ruta = Path(a.config)
    if not ruta.is_absolute():
        ruta = C.RAIZ_REPO / ruta
    cfg = C.cargar(ruta)
    print('[config] %s' % cfg['_archivo'])
    return cfg
