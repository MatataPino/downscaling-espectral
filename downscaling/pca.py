# -*- coding: utf-8 -*-
"""Reduccion de dimensionalidad del registro espectral del nodo oceanico.

Dos bases distintas, con propositos distintos:

  base_centrada       EOF de los espectros solo centrados. La usa la seleccion
                      por K-medias. (programa original 01)
  pca_estandarizado   EOF de los espectros estandarizados celda a celda, como
                      en Camus et al. (2013). Define el espacio en que se
                      selecciona por maxima disimilitud y en que interpola el
                      RBF. (programa original 29)

Ambas recorren el registro por bloques, sin cargarlo entero en memoria. El
tamano de bloque forma parte del resultado a nivel de bit: con otro tamano las
sumas se acumulan en otro orden y los valores cambian en el ultimo digito.
"""
import numpy as np


def base_centrada(Spec, frec, dirs, bloque, n_modos):
    Nt, Ndir, Nf = Spec.shape; D = Ndir*Nf
    ssum = np.zeros(D)
    for i in range(0, Nt, bloque):
        blk = np.array(Spec[i:i+bloque]).reshape(-1, D)
        ssum += blk.sum(0)
    mean = ssum / Nt
    Cov = np.zeros((D, D))
    for i in range(0, Nt, bloque):
        blk = np.array(Spec[i:i+bloque]).reshape(-1, D) - mean
        Cov += blk.T @ blk
    Cov /= (Nt - 1)
    w, V = np.linalg.eigh(Cov)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    vr = w / w.sum()
    return dict(mean=mean, EOF=V[:, :n_modos].T, var_ratio=vr[:n_modos],
                frec=frec, dirs=dirs, Ndir=Ndir, Nf=Nf, Nt=Nt)


def pca_estandarizado(Spec, frec, dirs, bloque, d):
    """Devuelve la base estandarizada, los componentes de todo el registro y los
    parametros integrados de cada estado (Hs, Tp, direccion de pico)."""
    Ndir, Nf = len(dirs), len(frec); D = Ndir*Nf; Nt = Spec.shape[0]
    df = np.gradient(frec); dth = np.deg2rad(np.gradient(dirs))
    W = df[None, None, :]*dth[:, None][None, :, :]

    # pasada 1: momentos para media, desviacion y correlacion, mas parametros
    S1 = np.zeros(D); S2 = np.zeros((D, D))
    Hs = np.empty(Nt); Tp = np.empty(Nt); Dp = np.empty(Nt)
    for i in range(0, Nt, bloque):
        raw = np.array(Spec[i:i+bloque]); X = raw.reshape(-1, D)
        S1 += X.sum(0); S2 += X.T@X
        Hs[i:i+bloque] = 4*np.sqrt(np.clip((raw*W).sum((1, 2)), 0, None))
        Sf = (raw*dth[:, None][None, :, :]).sum(1); Tp[i:i+bloque] = 1.0/frec[Sf.argmax(1)]
        E = (raw*df[None, None, :]).sum(2); i0 = E.argmax(1)
        ix = np.stack([(i0-1) % Ndir, i0, (i0+1) % Ndir], 1)
        w = np.take_along_axis(E, ix, 1); th = np.deg2rad(dirs[ix])
        Dp[i:i+bloque] = np.rad2deg(np.arctan2((w*np.sin(th)).sum(1), (w*np.cos(th)).sum(1))) % 360
    mu = S1/Nt
    Cov = S2/Nt-np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(Cov), 0, None))
    val = sd > 1e-12                         # celdas con varianza; el resto no informa

    Corr = Cov[np.ix_(val, val)]/np.outer(sd[val], sd[val])
    ev, EV = np.linalg.eigh(Corr)
    o = np.argsort(ev)[::-1]; ev = ev[o]; EV = EV[:, o]
    vexp = 100*np.cumsum(ev)/ev.sum()
    EOFs = EV.T

    # pasada 2: componentes principales de todos los estados
    E_use = EOFs[:d]
    Cc = np.empty((Nt, d), np.float32)
    for i in range(0, Nt, bloque):
        X = np.array(Spec[i:i+bloque]).reshape(-1, D)
        Cc[i:i+bloque] = ((X[:, val]-mu[val])/sd[val])@E_use.T

    return dict(mu=mu, sd=sd, val=val, EOF=E_use, EOFs_todos=EOFs, vexp=vexp,
                C=Cc, Hs=Hs, Tp=Tp, Dp=Dp, W=W, df=df)


def tabla_error(Spec, frec, dirs, base, lista_d, n_muestra=8000, semilla=0):
    """Error de reconstruccion de Hs, Tm01 y direccion media segun d.

    Es el criterio de Camus et al. (2013) para elegir el numero de componentes:
    se reconstruye una muestra de espectros con d modos y se mide cuanto se
    alejan sus parametros de los originales, en el clima operacional.
    """
    Ndir, Nf = len(dirs), len(frec); D = Ndir*Nf; Nt = Spec.shape[0]
    W = base['W']; df = base['df']; mu = base['mu']; sd = base['sd']; val = base['val']
    EOFs = base['EOFs_todos']; vexp = base['vexp']
    rng = np.random.default_rng(semilla); smp = np.sort(rng.choice(Nt, n_muestra, replace=False))
    raws = np.stack([Spec[i] for i in smp]); Xs = raws.reshape(len(smp), D)
    Zs = (Xs[:, val]-mu[val])/sd[val]

    def par_of(Xflat):
        S = np.maximum(Xflat, 0.).reshape(-1, Ndir, Nf)
        m0 = (S*W).sum((1, 2)); m1 = (S*W*frec[None, None, :]).sum((1, 2))
        E = (S*df[None, None, :]).sum(2); th = np.deg2rad(dirs)
        thm = np.rad2deg(np.arctan2((E*np.sin(th)[None, :]).sum(1), (E*np.cos(th)[None, :]).sum(1))) % 360
        return 4*np.sqrt(np.clip(m0, 0, None)), np.where(m1 > 0, m0/np.maximum(m1, 1e-30), np.nan), thm

    hs0, tm0, th0 = par_of(Xs); op = hs0 < 3.0
    filas = []
    for d in lista_d:
        if d > len(vexp): continue
        rec = np.zeros_like(Xs); rec[:, val] = (Zs@EOFs[:d].T)@EOFs[:d]*sd[val]+mu[val]
        h1, t1, d1 = par_of(rec)
        e = lambda a, b: np.sqrt((np.abs(a-b)[op]**2).mean())
        ed = np.sqrt((np.abs((d1-th0+180) % 360-180)[op]**2).mean())
        filas.append((d, vexp[d-1], e(h1, hs0), e(t1, tm0), ed))
    return filas
