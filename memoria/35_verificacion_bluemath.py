# -*- coding: utf-8 -*-
"""Verificacion cruzada contra BlueMath_tk, la implementacion de referencia del
grupo GeoOcean (Universidad de Cantabria), que es el grupo de Camus y Mendez.

No sustituye a nuestro codigo: lo contrasta. Si ambos coinciden, se puede
afirmar en la memoria que la implementacion propia reproduce la del grupo que
publico el metodo; si no coinciden, hay un error que conviene encontrar antes.

Se comparan las dos piezas que deciden el resultado:
  (1) RBF  : interpolacion gaussiana con sigma optimizado por Rippa
  (2) MDA  : seleccion por maxima disimilitud

pip install bluemath-tk   (GPL-3.0)
"""
import os, csv, time, numpy as np, pandas as pd, h5py
from scipy.interpolate import RBFInterpolator
BASE=r"RAIZ_PROYECTO"
PCA=BASE+"/Actividad 2/analisis con PCA"; CAS=BASE+"/Actividad 2/SWAN/casos_camus"
APM=BASE+"/Programa SWAN/AP_San_Vicente.mat"
DIRTAB=BASE+"/Actividad 2/SWAN/output_adcp_unstruc.tab"
ADCPCSV=r"RUTA_DATOS/Series de Tiempo Olas San Vicente(Series de Tiempo).csv"

K=np.load(PCA+"/datos/camus_std_d30.npz")
mu=K['mu']; sd=K['sd']; val=K['val']; EOF=K['EOF']; lo=K['lo']; hi=K['hi']; d=int(K['d'])
meta=np.load(CAS+"/meta.npy")
nrm=lambda C:(C-lo)/(hi-lo+1e-12)

def rpar(p):
    for ln in open(p,encoding='latin-1'):
        if ln.startswith('%') or not ln.strip(): continue
        v=ln.split(); return float(v[0]),float(v[2]),float(v[3])

# ---------- datos de entrenamiento en el ADCP ----------
f=h5py.File(BASE+"/Actividad 1/NID_027.mat",'r'); Spec=f['NID_027']['Spec']
X,Y=[],[]
for k in range(len(meta)):
    p=f"{CAS}/adcp_{k:03d}.tab"
    if not os.path.isfile(p): continue
    hs,tp,di=rpar(p)
    raw=np.array(Spec[int(meta[k,0])]).reshape(-1)
    X.append(((raw[val]-mu[val])/sd[val])@EOF.T); Y.append([hs,tp,di])
X=np.array(X); Y=np.array(Y)
_,uq=np.unique(np.round(X,6),axis=0,return_index=True); uq=np.sort(uq); X,Y=X[uq],Y[uq]
Xn=nrm(X)
print("[0] entrenamiento: %d casos en %dD"%(len(Xn),d))

# ---------- serie del periodo ADCP ----------
fa=h5py.File(APM,'r')['NID_027']; rawA=np.array(fa['Spec']); tA=np.array(fa['time'])
Fa=rawA.reshape(len(rawA),-1)
Cq=nrm(((Fa[:,val]-mu[val])/sd[val])@EOF.T)
t_rec=np.array([np.datetime64('%04d-%02d-%02dT%02d:00'%(tA[0,i],tA[1,i],tA[2,i],tA[3,i]))
                for i in range(tA.shape[1])])
rows=[r for r in csv.reader(open(ADCPCSV,encoding='latin-1'))][1:]
tm=[];Hm=[];Tmm=[];Dm=[]
for r in rows:
    if len(r)<7 or not r[0].strip(): continue
    tm.append(np.datetime64('%04d-%02d-%02dT%02d:00'%(int(r[0]),int(r[1]),int(r[2]),int(r[3]))))
    Hm.append(float(r[4]));Tmm.append(float(r[5]));Dm.append((float(r[6])-7.5)%360)
tm=np.array(tm);Hm=np.array(Hm);Tmm=np.array(Tmm);Dm=np.array(Dm)
dr=[l.split() for l in open(DIRTAB) if not l.startswith('%') and l.strip()]
Hd=np.array([float(x[0]) for x in dr])
td=np.datetime64('2025-08-15T18:00')+np.arange(len(Hd))*np.timedelta64(3,'h')
at=lambda T,V,q:np.array([dict(zip(T,V)).get(x,np.nan) for x in q])
Hsm=at(tm,Hm,td);Tpm=at(tm,Tmm,td);Dirm=at(tm,Dm,td)
sw=(Tpm>10)&(Dirm>=220)&(Dirm<=320)&np.isfinite(Hsm)&np.isfinite(Hd)

# ---------- (1) nuestro RBF gaussiano + Rippa ----------
def g_sys(Xn,c):
    M,dd_=Xn.shape
    R2=((Xn[:,None,:]-Xn[None,:,:])**2).sum(2)
    return np.block([[np.exp(-R2/(2*c*c)),np.c_[np.ones(M),Xn]],
                     [np.c_[np.ones(M),Xn].T,np.zeros((dd_+1,dd_+1))]])
def g_loo(Xn,y,c):
    M=len(y); Kk=g_sys(Xn,c)
    try:
        sol=np.linalg.solve(Kk,np.r_[y,np.zeros(Xn.shape[1]+1)]); Ki=np.linalg.inv(Kk)
    except np.linalg.LinAlgError: return np.inf
    dg=np.diag(Ki)[:M]
    if np.any(np.abs(dg)<1e-14): return np.inf
    return np.sqrt(np.mean((sol[:M]/dg)**2))
def g_pred(Xn,y,c,Q):
    M=len(Xn); sol=np.linalg.solve(g_sys(Xn,c),np.r_[y,np.zeros(Xn.shape[1]+1)])
    R2=((Q[:,None,:]-Xn[None,:,:])**2).sum(2)
    return np.exp(-R2/(2*c*c))@sol[:M]+np.c_[np.ones(len(Q)),Q]@sol[M:]
cs=np.logspace(-1.5,1.5,60)
ls=np.array([g_loo(Xn,Y[:,0],c) for c in cs])
c_ours=float(cs[int(np.nanargmin(ls))])
p_ours=g_pred(Xn,Y[:,0],c_ours,Cq)
print("[1] NUESTRO  : sigma=%.4f  LOO=%.4f m"%(c_ours,np.nanmin(ls)))

# ---------- (2) RBF de BlueMath_tk ----------
from bluemath_tk.interpolation.rbf import RBF
cols=["pc%02d"%i for i in range(d)]
sub=pd.DataFrame(Xn,columns=cols); tgt=pd.DataFrame({"Hs":Y[:,0]})
qq =pd.DataFrame(Cq,columns=cols)
t0=time.time()
rbf=RBF(sigma_min=0.05,sigma_max=3.0,sigma_diff=0.01,kernel="gaussian",smooth=0.0)
rbf.fit(subset_data=sub,target_data=tgt,normalize_target_data=True)
p_bm=rbf.predict(dataset=qq)["Hs"].values
sig=getattr(rbf,"sigma_opt",None)
print("[2] BLUEMATH : sigma=%s  (%.0fs)"%(("%.4f"%sig) if sig else "?",time.time()-t0))

# ---------- comparacion ----------
def met(o,m):
    o=o[sw];m=m[sw];k=np.isfinite(o)&np.isfinite(m);o=o[k];m=m[k]
    r=np.corrcoef(o,m)[0,1]
    return r**2,(m-o).mean(),np.sqrt(((m-o)**2).mean())
h_o=at(t_rec,p_ours,td); h_b=at(t_rec,p_bm,td)
print("\n=== RBF: validacion en el ADCP (filtro swell, N=%d) ==="%sw.sum())
print("%-14s %7s %8s %8s   %s"%("implementacion","R2dir","bias","RMSE","rango"))
for lbl,h in (("nuestra",h_o),("BlueMath",h_b)):
    r2,b,rm=met(Hd,h)
    print("%-14s %7.4f %+8.4f %8.4f   %.3f a %.3f m"%(lbl,r2,b,rm,np.nanmin(h[sw]),np.nanmax(h[sw])))
dif=np.abs(p_ours-p_bm)
print("\ndiferencia punto a punto sobre los %d estados del periodo:"%len(p_ours))
print("   media %.2e m | mediana %.2e m | max %.2e m"%(dif.mean(),np.median(dif),dif.max()))
print("   correlacion entre ambas series: %.8f"%np.corrcoef(p_ours,p_bm)[0,1])
np.savez(PCA+"/datos/verificacion_bluemath.npz",p_ours=p_ours,p_bluemath=p_bm,
         sigma_ours=c_ours,sigma_bluemath=sig if sig else np.nan)
print("\n[3] guardado -> datos/verificacion_bluemath.npz")
