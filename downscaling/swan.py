# -*- coding: utf-8 -*-
"""Generacion y ejecucion de los casos estacionarios de SWAN.

escribir_casos  espectro de contorno y archivo de entrada de cada caso.
                (programa original 30)
correr          propagacion en paralelo, reanudable. (programa original 31)

Unidades: el reanalisis entrega la densidad en m2/Hz/rad con las direcciones en
la primera dimension; SWAN la espera en m2/Hz/grado como matriz frecuencia x
direccion, de modo que se traspone y se multiplica por pi/180.

El espectro se replica en un fichero por lado de contorno, porque SWAN no admite
abrir el mismo archivo en mas de un contorno.

Rutas dentro del archivo de entrada: relativas a la raiz de SWAN (la carpeta que
contiene la malla y las carpetas de casos). Cada proceso de correr() trabaja en
un subdirectorio de esa raiz, asi que al ejecutar se les antepone '../'.
"""
import os, re, time, shutil, subprocess, queue, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from . import config as C
from .espectros import tab_con_datos


def _cabecera_sp2(lon, lat, frec, dirs, comentario):
    nf, nd = len(frec), len(dirs)
    return ("SWAN   1                                Swan standard spectral file, version\n"
            "$ %s\n" % comentario +
            "LONLAT                                  locations in spherical coordinates\n"
            "     1                                  number of locations\n"
            "  %.3f  %.3f\n" % (lon, lat) +
            "AFREQ                                   absolute frequencies in Hz\n"
            "    %d                                  number of frequencies\n" % nf
            + "".join("  %.4E\n" % v for v in frec) +
            "NDIR                                    spectral nautical directions in degr\n"
            "    %d                                  number of directions\n" % nd
            + "".join("  %.4f\n" % v for v in dirs) +
            "QUANT\n     1                                  number of quantities in table\n"
            "VaDens                                  variance densities in m2/Hz/degr\n"
            "m2/Hz/degr\n  -0.9900E+02                           exception value\n")


def cuerpo_sp2(raw, factor):
    """Bloque de datos de un espectro del nodo, ya en m2/Hz/grado."""
    nf = raw.shape[1]
    M = raw.T*(np.pi/180.0)
    return "FACTOR\n  %.4E\n" % factor + "".join(
        "  ".join("%.4E" % v for v in (M[r]/factor))+"\n" for r in range(nf))


def nombre_salida(punto, k):
    return "%s_%03d" % (punto.lower(), k)


def entrada(cfg, conjunto, k):
    """Texto del archivo de entrada de SWAN para el caso k de un conjunto."""
    sw = cfg['swan']; car = C.carpeta_casos(cfg, conjunto); n = "%03d" % k
    L = ["$ SWAN estacionario -- downscaling, conjunto %s, caso %s" % (conjunto, n),
         "PROJECT '%s' '%s'" % (sw['proyecto'], n),
         sw['preambulo'].strip(),
         "CGRID UNSTRUCTURED %s" % sw['cgrid'],
         "READGRID UNSTRUCTURED TRIANGLE '%s'" % sw['malla'],
         "INPGRID BOTTOM UNSTRUCTURED",
         "READINP BOTTOM 1.0 '%s.bot' FREE" % sw['malla'],
         "SET EXCMARK %d" % sw['excmark']]
    for lado in sw['lados_contorno']:
        L.append("BOUNDSPEC SIDE %d CONSTANT FILE '%s/%s_lado%d.sp2'" % (lado, car, n, lado))
    L.append(sw['fisica'].strip())
    for p in sw['puntos']:
        s = nombre_salida(p['nombre'], k)
        L.append("POINTS '%s' %s %s" % (p['nombre'], p['lon'], p['lat']))
        L.append("TABLE '%s' HEAD '%s/%s.tab' HS RTP TPS DIR DSPR TM01 TM02" % (p['nombre'], car, s))
        L.append("SPECOUT '%s' SPEC2D ABS '%s/%s.spc'" % (p['nombre'], car, s))
    L += ["COMPUTE STATIONARY", "STOP"]
    return "\n".join(L) + "\n"


def escribir_casos(cfg, conjunto, indices, Spec, frec, dirs, params=None, reportar=print):
    """Escribe los espectros de contorno y el archivo de entrada de cada caso."""
    raiz = C.raiz_swan(cfg); car = raiz / C.carpeta_casos(cfg, conjunto)
    car.mkdir(parents=True, exist_ok=True)
    nodo = cfg['nodo']; factor = cfg['swan']['factor_sp2']
    cab = _cabecera_sp2(nodo['lon'], nodo['lat'], frec, dirs, "downscaling: conjunto %s" % conjunto)
    for k, idx in enumerate(indices):
        txt = cab + cuerpo_sp2(np.array(Spec[int(idx)]), factor)
        for lado in cfg['swan']['lados_contorno']:
            open(car / ("%03d_lado%d.sp2" % (k, lado)), "w", newline="\n").write(txt)
        open(car / ("INPUT_%03d" % k), "w", newline="\n").write(entrada(cfg, conjunto, k))
    meta = np.asarray(indices)[:, None] if params is None else np.c_[indices, params]
    np.save(car / "meta.npy", meta)
    reportar("    %s: %d casos en %s" % (conjunto, len(indices), car))
    return car


# ------------------------------------------------------------------ ejecucion
_CMD_RUTA = re.compile(r"^(READGRID|READINP|BOUNDSPEC|TABLE|SPECOUT)\b")


def _rutas_a_padre(txt):
    """Antepone '../' a las rutas relativas de las ordenes que leen o escriben."""
    out = []
    for ln in txt.splitlines():
        if _CMD_RUTA.match(ln.strip()):
            ln = re.sub(r"'([^']+)'",
                        lambda m: "'%s'" % m.group(1) if os.path.isabs(m.group(1)) or m.group(1).startswith('../')
                        else "'../%s'" % m.group(1), ln)
        out.append(ln)
    return "\n".join(out) + "\n"


def correr(cfg, conjunto, reportar=print):
    """Propaga todos los casos pendientes de un conjunto, en procesos simultaneos.

    Cada proceso trabaja en su propio directorio, porque SWAN escribe PRINT,
    norm_end y Errfile en el directorio de trabajo y dos corridas se pisarian.
    Los directorios se toman de una cola y se devuelven al terminar: atarlos al
    indice del caso seria incorrecto con mas casos que procesos.
    Es reanudable: se saltan los casos cuya salida ya tiene datos.
    """
    sw = cfg['swan']; raiz = C.raiz_swan(cfg); car = raiz / C.carpeta_casos(cfg, conjunto)
    exe = str(C.ejecutable_swan(cfg)); nw = int(sw['procesos']); timeout = int(sw['timeout_s'])
    p0 = sw['puntos'][0]['nombre']
    casos = sorted(f.split("_")[1] for f in os.listdir(car) if f.startswith("INPUT_"))
    hecho = lambda n: tab_con_datos(car / (nombre_salida(p0, int(n)) + ".tab"))
    pend = [n for n in casos if not hecho(n)]
    reportar("    %s: %d casos | %d pendientes | %d procesos" % (conjunto, len(casos), len(pend), nw))
    if not pend:
        return 0, 0

    libres = queue.Queue()
    for j in range(nw):
        d = raiz / ("_w%d" % j); d.mkdir(exist_ok=True)
        ini = raiz / "swaninit"
        if ini.is_file() and not (d / "swaninit").is_file():
            shutil.copy2(ini, d / "swaninit")
        libres.put(d)
    lock = threading.Lock(); t0 = time.time(); est = {"ok": 0, "fail": 0}

    def uno(n):
        wd = libres.get()
        try:
            for f in ('PRINT', 'norm_end', 'Errfile'):
                try: (wd / f).unlink()
                except OSError: pass
            txt = open(car / ("INPUT_%s" % n), encoding='latin-1').read()
            open(wd / "INPUT", "w", newline="\n").write(_rutas_a_padre(txt))
            try:
                subprocess.run([exe], cwd=wd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            ne = wd / 'norm_end'
            bien = ne.is_file() and 'Normal end' in open(ne, encoding='latin-1', errors='replace').read()
        finally:
            libres.put(wd)
        st = 'OK' if (bien and hecho(n)) else 'FAIL'
        with lock:
            est["ok" if st == 'OK' else "fail"] += 1
            hechos = est["ok"]+est["fail"]; el = time.time()-t0
            open(car / "progreso.txt", "w").write(
                "caso %d/%d  OK=%d FAIL=%d  transcurrido %.1f min  ETA %.1f min\n"
                % (hechos, len(pend), est["ok"], est["fail"], el/60, el/hechos*(len(pend)-hechos)/60))
        return st

    with ThreadPoolExecutor(max_workers=nw) as ex:
        list(ex.map(uno, pend))
    reportar("    %s: %d OK, %d fallidos en %.1f min" % (conjunto, est["ok"], est["fail"], (time.time()-t0)/60))
    return est["ok"], est["fail"]
