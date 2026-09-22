# -*- coding: utf-8 -*-
"""Metodologia de Camus et al. (2013) aplicada al espectro, sin adaptaciones.

Diferencia con lo que se venia haciendo: aqui las variables se ESTANDARIZAN
antes del PCA (media cero, desviacion uno en cada celda del espectro), tal
como hace el paper con sus 35 parametros, "para evitar problemas debidos a
las diferentes escalas". Con ello el espacio reducido deja de estar dominado
por la energia y el MDA puede repartirse por todo el clima en lugar de
quedarse en el envolvente de alta energia.

Etapas, en el orden del paper:
  1. estandarizar las 696 densidades espectrales
  2. PCA -> d componentes (criterio: RMSE de reconstruccion)
  3. MDA en el espacio EOF, M=500, primer elemento el de Hs maximo
  4. los casos seleccionados NO se proyectan de vuelta: se identifican las
     fechas y se propagan los espectros ORIGINALES

Este script cubre 1-3. No lanza ninguna corrida de SWAN.
"""
import time, numpy as np, h5py
BASE=r"RAIZ_PROYECTO"
PCA=BASE+"/Actividad 2/analisis con PCA"; OUTD=PCA+"/datos"
NODE=BASE+"/Actividad 1/NID_027.mat"
M_SEL=500; CH=20000; t0=time.time()

Z=np.load(OUTD+"/eof_nodo_completo.npz")
frec=Z['frec'].ravel(); dirs=Z['dirs'].ravel()
Ndir=int(Z['Ndir']); Nf=int(Z['Nf']); D=Ndir*Nf
df=np.gradient(frec); dth=np.deg2rad(np.gradient(dirs))
W=df[None,None,:]*dth[:,None][None,:,:]

f=h5py.File(NODE,'r'); Spec=f['NID_027']['Spec']; Nt=Spec.shape[0]
print("[0] %d espectros de %d celdas (%d dir x %d frec)"%(Nt,D,Ndir,Nf))

# --- pasada 1: momentos para media/desviacion y matriz de correlacion -------
S1=np.zeros(D); S2=np.zeros((D,D))
Hs=np.empty(Nt); Tp=np.empty(Nt); Dp=np.empty(Nt)
for i in range(0,Nt,CH):
    raw=np.array(Spec[i:i+CH]); X=raw.reshape(-1,D)
    S1+=X.sum(0); S2+=X.T@X
    Hs[i:i+CH]=4*np.sqrt(np.clip((raw*W).sum((1,2)),0,None))
    Sf=(raw*dth[:,None][None,:,:]).sum(1); Tp[i:i+CH]=1.0/frec[Sf.argmax(1)]
    E=(raw*df[None,None,:]).sum(2); i0=E.argmax(1)
    ix=np.stack([(i0-1)%Ndir,i0,(i0+1)%Ndir],1)
    w=np.take_along_axis(E,ix,1); th=np.deg2rad(dirs[ix])
    Dp[i:i+CH]=np.rad2deg(np.arctan2((w*np.sin(th)).sum(1),(w*np.cos(th)).sum(1)))%360
mu=S1/Nt
Cov=S2/Nt-np.outer(mu,mu)
sd=np.sqrt(np.clip(np.diag(Cov),0,None))
val=sd>1e-12                      # celdas con varianza; el resto no informa
print("[1] celdas con varianza: %d de %d   (t=%ds)"%(val.sum(),D,time.time()-t0))

sdv=sd.copy(); sdv[~val]=1.0
Corr=Cov[np.ix_(val,val)]/np.outer(sd[val],sd[val])
ev,EV=np.linalg.eigh(Corr)
o=np.argsort(ev)[::-1]; ev=ev[o]; EV=EV[:,o]
vexp=100*np.cumsum(ev)/ev.sum()
EOFs=EV.T                          # (nval, nval), filas = modos
print("[2] PCA estandarizado: varianza acumulada")
for d in (5,10,13,16,20,30,50,80,120):
    if d<=len(ev): print("      d=%3d -> %.2f%%"%(d,vexp[d-1]))

# criterio de Camus: d tal que el error de reconstruccion sea aceptable.
# Se evalua sobre una muestra, reconstruyendo el espectro y sus parametros.
rng=np.random.default_rng(0); smp=np.sort(rng.choice(Nt,8000,replace=False))
raws=np.stack([Spec[i] for i in smp]); Xs=raws.reshape(len(smp),D)
Zs=(Xs[:,val]-mu[val])/sd[val]
def par_of(Xflat):
    S=np.maximum(Xflat,0.).reshape(-1,Ndir,Nf)
    m0=(S*W).sum((1,2)); m1=(S*W*frec[None,None,:]).sum((1,2))
    E=(S*df[None,None,:]).sum(2); th=np.deg2rad(dirs)
    thm=np.rad2deg(np.arctan2((E*np.sin(th)[None,:]).sum(1),(E*np.cos(th)[None,:]).sum(1)))%360
    return 4*np.sqrt(np.clip(m0,0,None)),np.where(m1>0,m0/np.maximum(m1,1e-30),np.nan),thm
hs0,tm0,th0=par_of(Xs); op=hs0<3.0
print("[3] criterio de Camus (RMSE de reconstruccion, clima operacional)")
print("      %4s %8s | %8s %9s %10s"%("d","var[%]","Hs [m]","Tm01 [s]","th_m [deg]"))
for d in (10,13,16,20,30,50,80,120):
    if d>len(ev): continue
    rec=np.zeros_like(Xs); rec[:,val]=(Zs@EOFs[:d].T)@EOFs[:d]*sd[val]+mu[val]
    h1,t1,d1=par_of(rec)
    e=lambda a,b:np.sqrt((np.abs(a-b)[op]**2).mean())
    ed=np.sqrt((np.abs((d1-th0+180)%360-180)[op]**2).mean())
    print("      %4d %8.2f | %8.3f %9.3f %10.2f"%(d,vexp[d-1],e(h1,hs0),e(t1,tm0),ed))

D_SEL=int(np.searchsorted(vexp,99.0)+1)
print("[4] d adoptado = %d (99%% de varianza, mismo umbral que Camus)"%D_SEL)

# --- pasada 2: componentes principales de todos los estados ----------------
E_use=EOFs[:D_SEL]
C=np.empty((Nt,D_SEL),np.float32)
for i in range(0,Nt,CH):
    X=np.array(Spec[i:i+CH]).reshape(-1,D)
    C[i:i+CH]=((X[:,val]-mu[val])/sd[val])@E_use.T
print("[5] PCs calculados (t=%ds)"%(time.time()-t0))

# --- MDA: un unico pase sobre toda la base, como el paper ------------------
lo=C.min(0); hi=C.max(0); Xn=((C-lo)/(hi-lo+1e-12)).astype(np.float32)
sel=[int(np.argmax(Hs))]                       # primer elemento: Hs maximo
dmin=np.sqrt(((Xn-Xn[sel[0]])**2).sum(1))
for _ in range(M_SEL-1):
    k=int(np.argmax(dmin)); sel.append(k)
    dmin=np.minimum(dmin,np.sqrt(((Xn-Xn[k])**2).sum(1)))
sel=np.array(sel)
print("[6] MDA M=%d listo (t=%.1f min)"%(M_SEL,(time.time()-t0)/60))

B=[(157.5,202.5,'S-SSW'),(202.5,247.5,'SW'),(247.5,292.5,'W'),(292.5,337.5,'NW')]
def resumen(s,tag):
    print("\n  --- %s (n=%d) ---"%(tag,len(s)))
    print("    Hs %.2f-%.2f m | Tp %.1f-%.1f s | Dp %.0f-%.0f deg"
          %(Hs[s].min(),Hs[s].max(),Tp[s].min(),Tp[s].max(),Dp[s].min(),Dp[s].max()))
    print("    operacionales (Hs<3): %d de %d"%((Hs[s]<3).sum(),len(s)))
    for a,c,n in B:
        m=(Dp>=a)&(Dp<c); ms=(Dp[s]>=a)&(Dp[s]<c)
        print("      %-6s clima %6.2f%%   casos %3d"%(n,100*m.mean(),ms.sum()))
    m=(Dp>=337.5)|(Dp<157.5); ms=(Dp[s]>=337.5)|(Dp[s]<157.5)
    print("      %-6s clima %6.2f%%   casos %3d"%('NNW-N',100*m.mean(),ms.sum()))
    print("    Tp>18 s: %d casos | Tp>20 s: %d casos"%((Tp[s]>18).sum(),(Tp[s]>20).sum()))
resumen(sel,"MDA 500, PCA estandarizado (Camus)")
resumen(sel[:150],"primeros 150 (comparable con la libreria vieja)")
print("\n  clima completo: Hs %.2f-%.2f | Tp %.1f-%.1f"%(Hs.min(),Hs.max(),Tp.min(),Tp.max()))
print("  operacional   : Hs %.2f-%.2f"%(Hs[Hs<3].min(),Hs[Hs<3].max()))

np.savez_compressed(OUTD+"/camus_estandarizado.npz",
    sel=sel,C=C[sel],Hs=Hs[sel],Tp=Tp[sel],Dp=Dp[sel],
    mu=mu,sd=sd,val=val,EOF=E_use,d=D_SEL,var_exp=vexp[:D_SEL],
    lo=lo,hi=hi,M=M_SEL,
    nota="PCA sobre espectros ESTANDARIZADOS (Camus 2013); MDA unico sobre toda la base")
print("\n[7] guardado -> datos/camus_estandarizado.npz  (%.1f min)"%((time.time()-t0)/60))
