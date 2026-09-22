# -*- coding: utf-8 -*-
"""Prueba de reproduccion: el pipeline, con la configuracion de San Vicente,
debe reproducir los resultados de la memoria.

Ejecuta los pasos 1 a 4 y 6 -el 5 es SWAN y no se vuelve a correr: el paso 6
usa las propagaciones existentes- y compara cada salida con el archivo que se
empleo en la memoria.

  Pasos 1 a 3 (base EOF, PCA + MDA, K-medias)    identicos BIT A BIT
  Paso 4 (entradas de SWAN)                        identicos, salvo 11 contornos de
                                                   K-medias escritos con otro FACTOR
                                                   de escala: misma densidad espectral
  Paso 6 (RBF y reconstruccion)                    a la precision reportada

Por que el paso 6 no se compara bit a bit: los espectros de SWAN en N4 con que
se entreno la reconstruccion de la memoria se sobrescribieron en una
propagacion posterior de los mismos casos, y SWAN no reproduce sus salidas bit a
bit entre corridas (diferencias de milimetros en Hs). Se exige entonces que
coincidan las cifras publicadas, a la precision con que se publican.

Uso:
    python tests/reproducir_san_vicente.py            # todos los pasos (~12 min)
    python tests/reproducir_san_vicente.py --hasta 4  # sin la reconstruccion
Requiere la variable de entorno MEMORIA apuntando a la carpeta de datos.
"""
import sys, argparse, copy, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from downscaling import config as C
from downscaling.espectros import abrir_nodo
from downscaling.pca import base_centrada, pca_estandarizado
from downscaling.seleccion import mda, kmedias
from downscaling.swan import escribir_casos
from downscaling.reconstruccion import reconstruir

# cifras publicadas en la memoria (verificacion con 60 estados, punto N4)
PUBLICADO = dict(Hs=(0.062, 0.120, 0.039, 0.926), Tp=(-0.330, 2.401, 0.729, 0.646),
                 Tp_dist=(5.0, 20.0, 31.7, 25.0, 16.7, 1.7),
                 Hs_pct=(0.886, 1.500, 2.415))

ap = argparse.ArgumentParser(); ap.add_argument('--hasta', type=int, default=6)
HASTA = ap.parse_args().hasta

cfg = C.cargar(C.RAIZ_REPO/'config'/'san_vicente.toml')
cfg['rutas']['salida'] = 'resultados/_prueba'
REF = C.datos(cfg, 'Actividad 2/analisis con PCA/datos')
REF_SW = C.datos(cfg, 'Actividad 2/SWAN')
res_ok = []


def informe(nombre, ok, detalle=''):
    res_ok.append(ok)
    print('  %-44s %s  %s' % (nombre, 'OK     ' if ok else 'DIFIERE', detalle))


def iguales(a, b, claves, nombre):
    for k in claves:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        ok = x.shape == y.shape and np.array_equal(x, y)
        informe('%s: %s' % (nombre, k), ok, '' if ok else 'forma %s / %s' % (x.shape, y.shape))


t0 = time.time()
f, g = abrir_nodo(cfg); Spec = g['Spec']; bloque = int(cfg['nodo']['bloque'])

print('\n[1] base EOF centrada')
B = base_centrada(Spec, np.array(g['frec']).ravel(), np.array(g['dir']).ravel(),
                  bloque, int(cfg['pca']['modos_base']))
iguales(B, np.load(REF/'eof_nodo_completo.npz'),
        ('mean', 'EOF', 'var_ratio', 'frec', 'dirs', 'Ndir', 'Nf', 'Nt'), 'base')
frec = B['frec'].ravel(); dirs = B['dirs'].ravel()

print('\n[2] PCA estandarizado y maxima disimilitud')
R = pca_estandarizado(Spec, frec, dirs, bloque, int(cfg['pca']['d']))
sel, lo, hi = mda(R['C'], R['Hs'], int(cfg['seleccion']['mda_casos']))
m = dict(sel=sel, C=R['C'][sel], Hs=R['Hs'][sel], Tp=R['Tp'][sel], Dp=R['Dp'][sel], mu=R['mu'],
         sd=R['sd'], val=R['val'], EOF=R['EOF'], lo=lo, hi=hi, d=R['EOF'].shape[0],
         M=int(cfg['seleccion']['mda_casos']))
iguales(m, np.load(REF/'camus_std_d30.npz'),
        ('sel', 'C', 'Hs', 'Tp', 'Dp', 'mu', 'sd', 'val', 'EOF', 'lo', 'hi', 'd', 'M'), 'mda')

print('\n[3] K-medias y verificacion')
s = cfg['seleccion']
km, _ = kmedias(Spec, B, bloque, int(s['kmedias_componentes']), float(s['kmedias_hs_max']),
                int(s['kmedias_casos']), s['kmedias_k_codo'], int(s['kmedias_muestra']),
                int(s['semilla']), reportar=lambda *a: None)
iguales(km, np.load(REF/'kmeans_casos.npz'),
        ('sel', 'C', 'Hs', 'Tp', 'Dp', 'inertia', 'Ks', 'Kopt', 'mu', 'sd'), 'kmedias')
ver = np.loadtxt(C.RAIZ_REPO/s['verificacion_indices'], dtype=np.int64, comments='#')
informe('verificacion: 60 indices', np.array_equal(ver, np.load(REF_SW/'casos_valid'/'meta.npy')))

if HASTA >= 4:
    print('\n[4] entradas de SWAN')
    c4 = copy.deepcopy(cfg); c4['swan']['raiz'] = str(C.salida(cfg, 'swan'))
    orig = dict(mda=('casos_camus', 'c', m['sel']), kmedias=('casos_kmeans', 'km_', km['sel']),
                verificacion=('casos_valid', 'v', ver))
    def densidad(lineas):
        """Espectro fisico: el archivo guarda enteros escalados por FACTOR."""
        i = lineas.index('FACTOR')
        fac = float(lineas[i+1].split()[0])
        return fac*np.array([float(x) for l in lineas[i+2:] for x in l.split()])

    for conj, (carp, pre, ids) in orig.items():
        car = escribir_casos(c4, conj, ids, Spec, frec, dirs, reportar=lambda *a: None)
        ident = 0; rel = 0.0
        for k in range(len(ids)):
            a = open(car/('%03d_lado2.sp2' % k)).read().splitlines()[2:]
            b = open(REF_SW/carp/('%s%03d_w.sp2' % (pre, k)), encoding='latin-1').read().splitlines()[2:]
            if a == b:
                ident += 1
            else:
                va, vb = densidad(a), densidad(b)
                rel = max(rel, np.abs(va-vb).max()/np.abs(vb).max())
        informe('contornos %s' % conj, ident == len(ids) or rel < 1e-4,
                '%d/%d identicos al caracter; el resto, misma densidad (dif. relativa %.0e)'
                % (ident, len(ids), rel) if ident < len(ids) else '%d/%d identicos' % (ident, len(ids)))
    omitir = ('$', 'PROJECT', 'BOUNDSPEC', 'POINTS', 'TABLE', 'SPECOUT')
    fis = lambda t: sorted(l.strip() for l in t.splitlines() if l.strip() and not l.strip().startswith(omitir))
    a = open(C.salida(cfg, 'swan')/'casos_mda'/'INPUT_000').read()
    b = open(REF_SW/'casos_camus'/'INPUT_c000', encoding='latin-1').read()
    informe('entrada de SWAN: ordenes de calculo', fis(a) == fis(b))
    informe("entrada de SWAN: punto N4", "POINTS 'N4' -73.16406 -36.73878" in a)

if HASTA >= 6:
    print('\n[6] RBF y reconstruccion (usa las propagaciones existentes; ~7 min)')
    c6 = copy.deepcopy(cfg); c6['swan']['raiz'] = str(REF_SW)
    c6['swan']['carpetas'] = dict(mda='casos_camus', kmedias='casos_kmeans', verificacion='casos_valid')
    r, v = reconstruir(c6, m['sel'], km['sel'], ver, m, B, reportar=lambda *a: None)
    np.savez_compressed(C.salida(cfg, 'reconstruccion_N4_prueba.npz'), **r)
    ref = np.load(REF/'reconstruccion_final_N4.npz')
    informe('casos de entrenamiento', int(r['M']) == int(ref['M']), '%d' % int(r['M']))
    informe('modos de salida', int(r['n_modos']) == int(ref['n_modos']), '%d' % int(r['n_modos']))
    informe('clima operacional', np.array_equal(r['operacional'], ref['operacional']))
    for n in ('Hs', 'Tp'):
        pub = PUBLICADO[n]; got = tuple(round(x, 3) for x in v[n])
        tol = 0.0021 if n == 'Tp' else 0.0005
        informe('verificacion %s (sesgo, RMSE, med|e|, r)' % n,
                all(abs(x-y) <= tol for x, y in zip(got, pub)), 'obtenido %s | publicado %s' % (got, pub))
    got = tuple(round(x, 1) for x in v['Tp_dist'][1])
    informe('distribucion de Tp reconstruida [%]', got == PUBLICADO['Tp_dist'], str(got))
    h = r['HsTpDir'][:, 0].astype(float); hr = ref['HsTpDir'][:, 0].astype(float)
    pct = tuple(round(x, 3) for x in np.percentile(h, [50, 90, 99]))
    informe('Hs p50 / p90 / p99 en N4', pct == PUBLICADO['Hs_pct'], str(pct))
    informe('Hs estado a estado, 407.592 horas', np.abs(h-hr).max() < 0.01,
            'max |dif| %.4f m, mediana %.4f m' % (np.abs(h-hr).max(), np.median(np.abs(h-hr))))

f.close()
n_ok = sum(res_ok)
print('\n%d de %d comprobaciones superadas en %.1f min'
      % (n_ok, len(res_ok), (time.time()-t0)/60))
print('RESULTADO: %s' % ('REPRODUCE LA MEMORIA' if n_ok == len(res_ok) else 'HAY DIFERENCIAS'))
sys.exit(0 if n_ok == len(res_ok) else 1)
