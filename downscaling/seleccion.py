# -*- coding: utf-8 -*-
"""Seleccion de los estados de mar que se propagan con SWAN.

  mda          maxima disimilitud (Kennard y Stone, 1969): cubre la frontera
               del espacio de estados. (programa original 29)
  kmedias      K-medias sobre el clima operacional, con medoides: cubre la
               densidad del regimen frecuente. (programa original 08)
  verificacion estados aleatorios del clima operacional, ajenos al
               entrenamiento, para medir el error en el interior del dominio.

Camus et al. (2011a) establecen que la maxima disimilitud es adecuada para
propagar y K-medias para representar el clima medio; aqui se combinan.
"""
import numpy as np
from scipy.cluster.vq import kmeans2


def mda(C, Hs, M):
    """Maxima disimilitud sobre componentes normalizadas a [0,1] por su rango.
    El primer elemento es el estado de mayor altura significativa."""
    lo = C.min(0); hi = C.max(0); Xn = ((C-lo)/(hi-lo+1e-12)).astype(np.float32)
    sel = [int(np.argmax(Hs))]
    dmin = np.sqrt(((Xn-Xn[sel[0]])**2).sum(1))
    for _ in range(M-1):
        k = int(np.argmax(dmin)); sel.append(k)
        dmin = np.minimum(dmin, np.sqrt(((Xn-Xn[k])**2).sum(1)))
    return np.array(sel), lo, hi


def kmedias(Spec, base_centrada, bloque, n_componentes, hs_max, K, lista_k,
            n_muestra, semilla, reportar=print):
    """K-medias sobre las primeras componentes de la base centrada, restringido a
    los estados con Hs < hs_max y con cada componente a varianza unitaria.
    Devuelve los MEDOIDES: el estado horario real mas proximo a cada centroide."""
    mean = base_centrada['mean']; EOF = base_centrada['EOF'][:n_componentes]
    frec = base_centrada['frec'].ravel(); dirs = base_centrada['dirs'].ravel()
    Ndir = int(base_centrada['Ndir']); Nf = int(base_centrada['Nf']); D = Ndir*Nf
    Nt = Spec.shape[0]
    df = np.gradient(frec); dth = np.deg2rad(np.gradient(dirs))
    Cc = np.empty((Nt, n_componentes)); Hs = np.empty(Nt); Tp = np.empty(Nt); Dp = np.empty(Nt)
    for i in range(0, Nt, bloque):
        raw = np.array(Spec[i:i+bloque]); Cc[i:i+bloque] = (raw.reshape(-1, D)-mean)@EOF.T
        Hs[i:i+bloque] = 4*np.sqrt((raw*df[None, None, :]*dth[:, None][None, :, :]).sum((1, 2)))
        Sf = raw.sum(1); Tp[i:i+bloque] = 1/frec[Sf.argmax(1)]; Sd = raw.sum(2); Dp[i:i+bloque] = dirs[Sd.argmax(1)]

    op = np.where(Hs < hs_max)[0]; Cop = Cc[op]
    mu = Cop.mean(0); sd = Cop.std(0); Cs = (Cop-mu)/sd
    reportar("    clima operacional (Hs < %.1f m): %d estados (%.0f %%)" % (hs_max, len(op), 100*len(op)/Nt))
    rng = np.random.default_rng(semilla); sub = rng.choice(len(Cs), n_muestra, replace=False); Csub = Cs[sub]

    inertia = []
    for k in lista_k:
        cen, lab = kmeans2(Csub, k, minit='++', seed=semilla, iter=25, missing='warn')
        inertia.append(float(((Csub-cen[lab])**2).sum(1).mean()))
        reportar("    K = %3d  error de cuantizacion %.3f" % (k, inertia[-1]))
    inertia = np.array(inertia)

    cen, lab = kmeans2(Csub, K, minit='++', seed=semilla, iter=50, missing='warn')
    medoides = []
    for c in range(K):
        dd = ((Cs-cen[c])**2).sum(1); medoides.append(int(op[np.argmin(dd)]))
    medoides = np.unique(medoides)
    return dict(sel=medoides, C=Cc[medoides], Hs=Hs[medoides], Tp=Tp[medoides], Dp=Dp[medoides],
                inertia=inertia, Ks=np.array(lista_k), Kopt=K, mu=mu, sd=sd), Hs


def verificacion(Hs_nodo, hs_max, n, excluir, semilla):
    """Estados aleatorios del clima operacional, excluyendo los de entrenamiento.
    Solo se usa cuando el archivo de configuracion no fija una lista."""
    cand = np.setdiff1d(np.where(Hs_nodo < hs_max)[0], excluir)
    rng = np.random.default_rng(semilla)
    return np.sort(rng.choice(cand, n, replace=False))
