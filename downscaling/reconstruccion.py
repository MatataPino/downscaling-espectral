# -*- coding: utf-8 -*-
"""Interpolacion y reconstruccion del registro en el punto de destino.
(programa original 39)

1. Reune los casos de entrenamiento (maxima disimilitud + K-medias), elimina
   duplicados por hora y proyecta su espectro de N10 en el espacio reducido.
2. Construye la base EOF de SALIDA sobre los espectros propagados por SWAN.
3. Entrena 64 interpoladores: 4 parametricos (Hs, Tp, cos, sin de la
   direccion), de referencia, y uno por coeficiente espectral.
4. Evalua los interpoladores en todo el registro y recompone el espectro:
       E = espectro medio + sum_k c_k EOF_k
   Los parametros se INTEGRAN del espectro reconstruido.
5. Verifica contra los estados de verificacion, ajenos al entrenamiento.
"""
import time
import numpy as np
from . import config as C
from .espectros import leer_spc, leer_tab, dir_pico, abrir_nodo
from .rbf import malla_sigma, rippa, resolver
from .swan import nombre_salida

REDONDEO_SIGMA = 6        # decimales con que se agrupan los objetivos por sigma


def _carga(car, ids, punto, Spec, mu, sd, val, EOF):
    X, P, S, I = [], [], [], []
    fq = dd = None
    for k, idx in enumerate(ids):
        base = car / nombre_salida(punto, k)
        fa, fb = base.with_suffix('.tab'), base.with_suffix('.spc')
        if not (fa.is_file() and fb.is_file()): continue
        fq, dd, s = leer_spc(fb)
        if np.isnan(s).any(): continue
        hs, tp = leer_tab(fa)
        raw = np.array(Spec[int(idx)]).reshape(-1)
        X.append(((raw[val]-mu[val])/sd[val])@EOF.T)
        P.append([hs, tp, dir_pico(fq, dd, s[0])]); S.append(s[0].reshape(-1)); I.append(int(idx))
    return np.array(X), np.array(P), np.array(S), np.array(I), fq, dd


def reconstruir(cfg, sel_mda, sel_kmedias, sel_verif, reduccion, base_centrada,
                reportar=print):
    t0 = time.time()
    rc = cfg['rbf']; pto = rc['punto']; raiz = C.raiz_swan(cfg)
    mu, sd, val, EOF, lo, hi = (reduccion[k] for k in ('mu', 'sd', 'val', 'EOF', 'lo', 'hi'))
    d = int(reduccion['d'])
    nrm = lambda Cq: (Cq-lo)/(hi-lo+1e-12)

    f, g = abrir_nodo(cfg)
    Spec = g['Spec']; Nt = Spec.shape[0]

    # 1. conjunto de entrenamiento
    Xm, Pm, Sm, Im, fq, dd = _carga(raiz/C.carpeta_casos(cfg, 'mda'), sel_mda, pto, Spec, mu, sd, val, EOF)
    Xk, Pk, Sk, Ik, _, _ = _carga(raiz/C.carpeta_casos(cfg, 'kmedias'), sel_kmedias, pto, Spec, mu, sd, val, EOF)
    reportar("    entrenamiento: %d MDA + %d K-medias" % (len(Xm), len(Xk)))
    X = np.vstack([Xm, Xk]); P = np.vstack([Pm, Pk]); S = np.vstack([Sm, Sk]); I = np.r_[Im, Ik]
    _, u = np.unique(I, return_index=True); u = np.sort(u)
    X, P, S = X[u], P[u], S[u]
    Xn = nrm(X); M = len(Xn); nf, nd = len(fq), len(dd)
    reportar("    %d casos tras eliminar duplicados" % M)

    # 2. base EOF de salida
    Ym = S.mean(0); A = S-Ym
    _, sv, Vt = np.linalg.svd(A, full_matrices=False); vr = np.cumsum((sv**2)/(sv**2).sum())
    ko = min(int(rc['modos_salida']), len(sv)); E = Vt[:ko]; Yc = A@E.T
    reportar("    base de salida: %d modos, %.4f %% de varianza" % (ko, 100*vr[ko-1]))

    # 3. sigma por Rippa y sistemas
    grid = malla_sigma(rc['sigma_min'], rc['sigma_max'], int(rc['n_sigma']))
    tv = np.c_[P[:, 0], P[:, 1], np.cos(np.deg2rad(P[:, 2])), np.sin(np.deg2rad(P[:, 2]))]
    sp_, _ = rippa(Xn, tv, grid)
    sig_par = dict(zip(["Hs", "Tp", "Dir_u", "Dir_v"], sp_))
    sig_spc, _ = rippa(Xn, Yc, grid)
    for nm, v in list(sig_par.items())+[("espectral min", sig_spc.min()), ("espectral max", sig_spc.max())]:
        if float(v) <= grid[1] or float(v) >= grid[-2]:
            reportar("    AVISO: sigma de %s (%.4f) pegado al borde del barrido" % (nm, float(v)))
    reportar("    sigma: Hs %.4f | Tp %.4f | espectrales %.4f - %.4f (mediana %.4f)"
             % (sig_par['Hs'], sig_par['Tp'], sig_spc.min(), sig_spc.max(), np.median(sig_spc)))
    sol_par = [resolver(Xn, sig_par[k], tv[:, j]) for j, k in enumerate(["Hs", "Tp", "Dir_u", "Dir_v"])]
    grp = {}
    for i in range(ko): grp.setdefault(round(sig_spc[i], REDONDEO_SIGMA), []).append(i)
    sol_spc = {c: resolver(Xn, c, Yc[:, ix]) for c, ix in grp.items()}

    # 4. reconstruccion del registro
    frN = base_centrada['frec'].ravel(); diN = base_centrada['dirs'].ravel()
    W10 = np.gradient(frN)[None, None, :]*np.deg2rad(np.gradient(diN))[:, None][None, :, :]
    dfq = np.abs(np.gradient(fq)); ddg = np.abs(np.gradient(dd)); lf = np.log(fq); thr = np.deg2rad(dd)
    CH = int(cfg['nodo']['bloque'])
    par = np.empty((Nt, 3), np.float32); esp = np.empty((Nt, 3), np.float32)
    Yca = np.empty((Nt, ko), np.float32); Tm = np.empty(Nt, np.float32); HsN = np.empty(Nt)
    for i in range(0, Nt, CH):
        raw = np.array(Spec[i:i+CH]); Xf = raw.reshape(len(raw), -1)
        HsN[i:i+CH] = 4*np.sqrt(np.clip((raw*W10).sum((1, 2)), 0, None))
        Q = nrm(((Xf[:, val]-mu[val])/sd[val])@EOF.T); Pq = np.c_[np.ones(len(Q)), Q]
        R2q = ((Q[:, None, :]-Xn[None, :, :])**2).sum(2)
        V = np.empty((len(Q), 4))
        for j, k in enumerate(["Hs", "Tp", "Dir_u", "Dir_v"]):
            c = sig_par[k]; s = sol_par[j]
            V[:, j] = (np.exp(-R2q/(2*c*c))@s[:M]+Pq@s[M:]).ravel()
        par[i:i+CH] = np.c_[V[:, 0], V[:, 1], np.rad2deg(np.arctan2(V[:, 3], V[:, 2])) % 360]
        co = np.empty((len(Q), ko))
        for c, ix in grp.items():
            s = sol_spc[c]; co[:, ix] = np.exp(-R2q/(2*c*c))@s[:M]+Pq@s[M:]
        Yca[i:i+CH] = co
        sp = np.maximum(co@E+Ym, 0.).reshape(-1, nf, nd); a = np.arange(len(sp))
        m0 = (sp*dfq[:, None][None, :, :]*ddg[None, None, :]).sum((1, 2))
        m1 = (sp*dfq[:, None][None, :, :]*ddg[None, None, :]*fq[None, :, None]).sum((1, 2))
        esp[i:i+CH, 0] = 4*np.sqrt(np.clip(m0, 0, None))
        Tm[i:i+CH] = np.where(m1 > 0, m0/np.maximum(m1, 1e-30), np.nan)
        # periodo de pico con ajuste parabolico en log(f) sobre las tres bandas vecinas
        Sf = (sp*ddg[None, None, :]).sum(2); j = Sf.argmax(1)
        jm = np.clip(j-1, 0, nf-1); jp = np.clip(j+1, 0, nf-1)
        y0, y1, y2 = Sf[a, jm], Sf[a, j], Sf[a, jp]; den = y0-2*y1+y2
        dl = np.clip(np.where(np.abs(den) > 1e-30, 0.5*(y0-y2)/np.where(np.abs(den) > 1e-30, den, 1), 0.), -.5, .5)
        esp[i:i+CH, 1] = 1.0/np.exp(lf[j]+dl*(lf[jp]-lf[jm])*0.5)
        Sd = (sp*dfq[None, :, None]).sum(1); i0 = Sd.argmax(1)
        ix = np.stack([(i0-1) % nd, i0, (i0+1) % nd], 1); w = np.take_along_axis(Sd, ix, 1); t_ = thr[ix]
        esp[i:i+CH, 2] = np.rad2deg(np.arctan2((w*np.sin(t_)).sum(1), (w*np.cos(t_)).sum(1))) % 360
        if (i//CH) % 5 == 0: reportar("      %d/%d  (%.1f min)" % (i, Nt, (time.time()-t0)/60))
    op = HsN < float(rc['hs_operacional'])

    # 5. verificacion
    ver = None
    if sel_verif is not None and len(sel_verif):
        car = raiz/C.carpeta_casos(cfg, 'verificacion'); TT = []
        for k, idx in enumerate(sel_verif):
            base = car/nombre_salida(pto, k)
            fa, fb = base.with_suffix('.tab'), base.with_suffix('.spc')
            if not (fa.is_file() and fb.is_file()): continue
            _, dd2, s = leer_spc(fb); hs, tp = leer_tab(fa)
            TT.append((int(idx), tp, hs, dir_pico(fq, dd2, s[0])))
        TT = np.array(TT); iv = TT[:, 0].astype(int)
        circ = lambda x, y: (x-y+180) % 360-180
        ver = {}
        for n, tr, pr, ang in (("Hs", TT[:, 2], esp[iv, 0], False), ("Tp", TT[:, 1], esp[iv, 1], False),
                               ("Dpeak", TT[:, 3], esp[iv, 2], True)):
            e = circ(pr, tr) if ang else pr-tr
            ver[n] = (e.mean(), np.sqrt((e**2).mean()), np.median(np.abs(e)), np.corrcoef(tr, pr)[0, 1])
        b = np.array([8, 10, 12, 14, 16, 18, 30])
        ver['Tp_dist'] = tuple(100*np.histogram(v, b)[0]/np.histogram(v, b)[0].sum()
                               for v in (TT[:, 1], esp[iv, 1]))
        ver['n'] = len(TT)

    pinfo = C.punto(cfg, pto)
    res = dict(lon=pinfo['lon'], lat=pinfo['lat'], utm_e=pinfo.get('utm_e', np.nan),
               utm_n=pinfo.get('utm_n', np.nan), depth_m=pinfo.get('profundidad', np.nan),
               HsTpDir=esp, HsTpDir_rbf=par, Tm01=Tm, Hs_spec=esp[:, 0], Ycoef_all=Yca,
               EOF_out=E, Yspec_mean=Ym, frec=fq, dirs=dd, nf=nf, nd=nd,
               time=np.array(g['time']), HsN10=HsN.astype(np.float32), operacional=op,
               d=d, M=M, n_modos=ko, sigmas_par=str(sig_par), sigmas_spc=sig_spc)
    f.close()
    reportar("    reconstruccion completa en %.1f min" % ((time.time()-t0)/60))
    return res, ver
