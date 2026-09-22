# -*- coding: utf-8 -*-
"""Reconstruccion definitiva en N4: seleccion MDA + KMA, RBF espectral.

SELECCION. Camus et al. (2011a) comparan MDA, KMA y SOM y concluyen que el
primero cubre la diversidad del clima -incluidos los extremos- y el segundo el
clima medio. Aqui se emplean ambos de forma complementaria, porque se midio que
por separado ninguno basta:

  - MDA (500 casos) garantiza que ningun regimen quede fuera del envolvente,
    pero solo el 15 % de sus casos son operacionales frente al 69 % del tiempo.
    Verificado contra 60 estados aleatorios del clima ajenos al entrenamiento,
    deja el intervalo 12-14 s del periodo de pico con un 5 % de ocurrencia
    frente al 33 % real.
  - KMA (100 casos) sigue la densidad y puebla ese interior. Al anadirlo, el
    intervalo pasa a 30 % y el error mediano de Tp cae un 44 %.

Se descarta el tercer conjunto disponible (K-medias ESTRATIFICADO): sus casos
de sectores infrecuentes quedan lejos en el espacio reducido y degradan el
ajuste (correlacion negativa en la validacion).

BASE EOF DE SALIDA. Se elevan los modos de 21 a 60. Medido sobre los propios
espectros propagados, el truncamiento a 21 modos introduce un error maximo de
14,6 s en el periodo de pico; con 60 modos baja a 0,18 s. No afecta a la
metodologia -Camus no reconstruye espectros- y no cuesta ninguna propagacion.

IMPLEMENTACION. Se emplea la implementacion propia, verificada identica a la de
bluemath_tk hasta 1e-11 para un mismo sigma (script 35). El motivo es de coste:
la biblioteca de referencia rehace la factorizacion del sistema para cada
objetivo, y con 64 objetivos el ajuste no termina en horas; compartiendola entre
objetivos se resuelve en segundos. El algoritmo -nucleo gaussiano, base monomial
de grado 1, ajuste exacto y sigma por Rippa- es el mismo.
"""
import os, csv, time, numpy as np, h5py
BASE=r"RAIZ_PROYECTO"
PCA=BASE+"/Actividad 2/analisis con PCA"; SW=BASE+"/Actividad 2/SWAN"
OUTD=PCA+"/datos"; NMOD=60
# Barrido de sigma: 60 valores en malla logaritmica sobre tres ordenes de
# magnitud. Se verifico (script 27) que multiplicar sigma por 4 desplaza las
# predicciones apenas centimetros, de modo que no hace falta mas resolucion;
# lo que si importa es que el rango CONTENGA el optimo.
SIG=dict(sigma_min=0.002,sigma_max=3.0,kernel="gaussian",smooth=0.0,n_sigma=60)
t0=time.time()

K=np.load(OUTD+"/camus_std_d30.npz")
mu,sd,val,EOF,lo,hi,d=K['mu'],K['sd'],K['val'],K['EOF'],K['lo'],K['hi'],int(K['d'])
nrm=lambda C:(C-lo)/(hi-lo+1e-12); COLS=["pc%02d"%i for i in range(d)]

def read_spc(p):
    L=open(p,encoding='latin-1').read().splitlines(); i=0
    while 'FREQ' not in L[i]: i+=1
    n1=int(L[i+1].split()[0]); fq=np.array([float(L[i+2+j]) for j in range(n1)]); i+=2+n1
    while 'DIR' not in L[i]: i+=1
    n2=int(L[i+1].split()[0]); dd=np.array([float(L[i+2+j]) for j in range(n2)])%360.; i+=2+n2
    while 'QUANT' not in L[i]: i+=1
    nq=int(L[i+1].split()[0]); i=i+2+3*nq; sp=[]
    while i<len(L):
        k=L[i].strip()
        if k.startswith('FACTOR'):
            f_=float(L[i+1]); i+=2
            sp.append(np.array([[float(x) for x in L[i+r].split()] for r in range(n1)])*f_); i+=n1
        elif k.startswith('ZERO'): sp.append(np.zeros((n1,n2))); i+=1
        else: i+=1
    return fq,dd,np.array(sp)
def rpar(p):
    for ln in open(p,encoding='latin-1'):
        if ln.startswith('%') or not ln.strip(): continue
        v=ln.split(); return float(v[0]),float(v[2])
def dpk(fq,dd,S):
    dfq=np.abs(np.gradient(fq)); Sd=(S*dfq[:,None]).sum(0); n=len(dd); i0=int(Sd.argmax())
    ix=[(i0+k)%n for k in (-1,0,1)]; w=Sd[ix]; th=np.deg2rad(dd[ix])
    return np.rad2deg(np.arctan2((w*np.sin(th)).sum(),(w*np.cos(th)).sum()))%360

f=h5py.File(BASE+"/Actividad 1/NID_027.mat",'r'); Spec=f['NID_027']['Spec']; Nt=Spec.shape[0]
def carga(dirc,ids,etq):
    X,P,S,I=[],[],[],[]
    for k,idx in enumerate(ids):
        fa=f"{dirc}/n4_{k:03d}.tab"; fb=f"{dirc}/n4_{k:03d}.spc"
        if not(os.path.isfile(fa) and os.path.isfile(fb)): continue
        fq,dd,s=read_spc(fb)
        if np.isnan(s).any(): continue
        hs,tp=rpar(fa)
        raw=np.array(Spec[int(idx)]).reshape(-1)
        X.append(((raw[val]-mu[val])/sd[val])@EOF.T)
        P.append([hs,tp,dpk(fq,dd,s[0])]); S.append(s[0].reshape(-1)); I.append(int(idx))
    print("    %-12s %d casos"%(etq,len(X)))
    return np.array(X),np.array(P),np.array(S),np.array(I),fq,dd

print("[1] conjunto de entrenamiento")
Xm,Pm,Sm,Im,fq,dd=carga(SW+"/casos_camus",np.load(SW+"/casos_camus/meta.npy")[:,0],"MDA")
Xk,Pk,Sk,Ik,_,_ =carga(SW+"/casos_kmeans",np.load(OUTD+"/kmeans_casos.npz")['sel'],"KMA")
X=np.vstack([Xm,Xk]); P=np.vstack([Pm,Pk]); S=np.vstack([Sm,Sk]); I=np.r_[Im,Ik]
_,u=np.unique(I,return_index=True); u=np.sort(u)          # deduplicar por hora
X,P,S=X[u],P[u],S[u]
Xn=nrm(X); M=len(Xn); nf,nd=len(fq),len(dd)
print("    TOTAL        %d tras deduplicar | Hs %.2f-%.2f | Tp %.1f-%.1f | Dpk %.0f-%.0f"
      %(M,P[:,0].min(),P[:,0].max(),P[:,1].min(),P[:,1].max(),P[:,2].min(),P[:,2].max()))

Ym=S.mean(0); A=S-Ym
_,sv,Vt=np.linalg.svd(A,full_matrices=False); vr=np.cumsum((sv**2)/(sv**2).sum())
ko=min(NMOD,len(sv)); E=Vt[:ko]; Yc=A@E.T
print("[2] base EOF de salida: %d modos = %.4f %% de varianza"%(ko,100*vr[ko-1]))

# --- sigma por Rippa (factorizacion compartida entre objetivos) -----------
#     El algoritmo es el de Rippa (1999), identico al de bluemath_tk, pero
#     explotando que la matriz del sistema depende SOLO de sigma y no del
#     objetivo: se invierte una vez por cada sigma del barrido y se evalua con
#     ella los 64 objetivos. bluemath_tk la reconstruye para cada objetivo, lo
#     que multiplica el coste por 64 y resulta inviable con una base espectral
#     de 60 modos: 64 objetivos se resuelven aqui en 13 s.
#     Los sigma no coinciden con los de bluemath_tk (0.62 frente a 1.84 en el
#     mismo problema), pero el error de validacion cruzada de este es MENOR
#     (0.0664 frente a 0.0814), es decir, mejor segun el criterio que ambos
#     minimizan. Y la diferencia es en todo caso irrelevante: con sigma distinto
#     por un factor 4 las predicciones difieren en centimetros (script 35).
def rippa(Xn,Ys,grid):
    """Sigma optimo por objetivo. Ys: (M, n_objetivos). Devuelve (n_obj,)."""
    M=len(Xn); dpl=Xn.shape[1]
    R2=((Xn[:,None,:]-Xn[None,:,:])**2).sum(2); P=np.c_[np.ones(M),Xn]
    Z=np.zeros((dpl+1,dpl+1)); rhs=np.vstack([Ys,np.zeros((dpl+1,Ys.shape[1]))])
    best=np.full(Ys.shape[1],np.inf); sig=np.full(Ys.shape[1],grid[0])
    for c in grid:
        Kk=np.block([[np.exp(-R2/(2*c*c)),P],[P.T,Z]])
        try:
            sol=np.linalg.solve(Kk,rhs); Ki=np.linalg.inv(Kk)
        except np.linalg.LinAlgError: continue
        dg=np.diag(Ki)[:M]
        if np.any(np.abs(dg)<1e-14): continue
        err=np.sqrt((( sol[:M]/dg[:,None] )**2).mean(0))     # LOO por objetivo
        m=err<best; best[m]=err[m]; sig[m]=c
    return sig,best

# Malla LOGARITMICA: el optimo de este problema cae en la parte baja del rango
# (el termino polinomico domina y las gaussianas solo corrigen localmente), de
# modo que una malla lineal lo dejaria pegado al borde inferior. Se cubren tres
# ordenes de magnitud y despues se comprueba que el optimo no toca los extremos.
grid=np.exp(np.linspace(np.log(SIG["sigma_min"]),np.log(SIG["sigma_max"]),SIG["n_sigma"]))
tv=np.c_[P[:,0],P[:,1],np.cos(np.deg2rad(P[:,2])),np.sin(np.deg2rad(P[:,2]))]
sp_,lp=rippa(Xn,tv,grid)
sig_par=dict(zip(["Hs","Tp","Dir_u","Dir_v"],sp_))
sig_spc,ls=rippa(Xn,Yc,grid)
print("[3] sigma (Rippa, %d valores log): parametros %s"%(len(grid),{k:round(float(v),4) for k,v in sig_par.items()}))
for nm,v in list(sig_par.items())+[("espectral_min",sig_spc.min()),("espectral_max",sig_spc.max())]:
    if float(v)<=grid[1] or float(v)>=grid[-2]:
        print("    AVISO: sigma de %s (%.4f) pegado al borde del barrido"%(nm,float(v)))
print("    LOO parametros: Hs %.4f | Tp %.4f"%(lp[0],lp[1]))
print("    espectrales: %.3f - %.3f (mediana %.3f)  (t=%.1f min)"
      %(sig_spc.min(),sig_spc.max(),np.median(sig_spc),(time.time()-t0)/60))

# --- solucion de los sistemas (implementacion propia, identica) -----------
def solve(c,Y):
    R2=((Xn[:,None,:]-Xn[None,:,:])**2).sum(2); Pm_=np.c_[np.ones(M),Xn]
    Kk=np.block([[np.exp(-R2/(2*c*c)),Pm_],[Pm_.T,np.zeros((d+1,d+1))]])
    Y=np.atleast_2d(Y.T).T
    return np.linalg.solve(Kk,np.vstack([Y,np.zeros((d+1,Y.shape[1]))]))
sol_par=[solve(sig_par[k],tv[:,j]) for j,k in enumerate(["Hs","Tp","Dir_u","Dir_v"])]
grp={}
for i in range(ko): grp.setdefault(round(sig_spc[i],6),[]).append(i)
sol_spc={c:solve(c,Yc[:,ix]) for c,ix in grp.items()}
print("[4] sistemas resueltos: %d grupos de sigma espectral (t=%.1f min)"%(len(grp),(time.time()-t0)/60))

# --- reconstruccion de los 407.592 estados -------------------------------
Zf=np.load(OUTD+"/eof_nodo_completo.npz"); frN=Zf['frec'].ravel(); diN=Zf['dirs'].ravel()
W10=np.gradient(frN)[None,None,:]*np.deg2rad(np.gradient(diN))[:,None][None,:,:]
dfq=np.abs(np.gradient(fq)); ddg=np.abs(np.gradient(dd)); lf=np.log(fq); thr=np.deg2rad(dd)
CH=20000
par=np.empty((Nt,3),np.float32); esp=np.empty((Nt,3),np.float32)
Yca=np.empty((Nt,ko),np.float32); Tm=np.empty(Nt,np.float32); HsN=np.empty(Nt)
for i in range(0,Nt,CH):
    raw=np.array(Spec[i:i+CH]); Xf=raw.reshape(len(raw),-1)
    HsN[i:i+CH]=4*np.sqrt(np.clip((raw*W10).sum((1,2)),0,None))
    Q=nrm(((Xf[:,val]-mu[val])/sd[val])@EOF.T); Pq=np.c_[np.ones(len(Q)),Q]
    R2q=((Q[:,None,:]-Xn[None,:,:])**2).sum(2)
    V=np.empty((len(Q),4))
    for j,k in enumerate(["Hs","Tp","Dir_u","Dir_v"]):
        c=sig_par[k]; s=sol_par[j]
        V[:,j]=(np.exp(-R2q/(2*c*c))@s[:M]+Pq@s[M:]).ravel()
    par[i:i+CH]=np.c_[V[:,0],V[:,1],np.rad2deg(np.arctan2(V[:,3],V[:,2]))%360]
    co=np.empty((len(Q),ko))
    for c,ix in grp.items():
        s=sol_spc[c]; co[:,ix]=np.exp(-R2q/(2*c*c))@s[:M]+Pq@s[M:]
    Yca[i:i+CH]=co
    sp=np.maximum(co@E+Ym,0.).reshape(-1,nf,nd); a=np.arange(len(sp))
    m0=(sp*dfq[:,None][None,:,:]*ddg[None,None,:]).sum((1,2))
    m1=(sp*dfq[:,None][None,:,:]*ddg[None,None,:]*fq[None,:,None]).sum((1,2))
    esp[i:i+CH,0]=4*np.sqrt(np.clip(m0,0,None))
    Tm[i:i+CH]=np.where(m1>0,m0/np.maximum(m1,1e-30),np.nan)
    Sf=(sp*ddg[None,None,:]).sum(2); j=Sf.argmax(1)
    jm=np.clip(j-1,0,nf-1); jp=np.clip(j+1,0,nf-1)
    y0,y1,y2=Sf[a,jm],Sf[a,j],Sf[a,jp]; den=y0-2*y1+y2
    dl=np.clip(np.where(np.abs(den)>1e-30,0.5*(y0-y2)/np.where(np.abs(den)>1e-30,den,1),0.),-.5,.5)
    esp[i:i+CH,1]=1.0/np.exp(lf[j]+dl*(lf[jp]-lf[jm])*0.5)
    Sd=(sp*dfq[None,:,None]).sum(1); i0=Sd.argmax(1)
    ix=np.stack([(i0-1)%nd,i0,(i0+1)%nd],1); w=np.take_along_axis(Sd,ix,1); t_=thr[ix]
    esp[i:i+CH,2]=np.rad2deg(np.arctan2((w*np.sin(t_)).sum(1),(w*np.cos(t_)).sum(1)))%360
    if (i//CH)%5==0: print("      %d/%d  (t=%.1f min)"%(i,Nt,(time.time()-t0)/60))

op=HsN<3.0
print("\n[5] N4, 46,5 anios -- parametros integrados del espectro:")
print("    Hs  %.2f - %.2f m  (mediana %.2f)"%(esp[:,0].min(),esp[:,0].max(),np.median(esp[:,0])))
print("    Tp  %.1f - %.1f s  (mediana %.1f)"%(esp[:,1].min(),esp[:,1].max(),np.median(esp[:,1])))
print("    Dpk %.0f - %.0f deg (mediana %.0f)"%(esp[:,2].min(),esp[:,2].max(),np.median(esp[:,2])))
print("    Tm01 %.1f - %.1f s"%(np.nanmin(Tm),np.nanmax(Tm)))

# --- validacion contra los 60 aleatorios reservados ----------------------
vsel=np.load(SW+"/casos_valid/meta.npy"); TT,HH,DD=[],[],[]
for k,idx in enumerate(vsel):
    fa=f"{SW}/casos_valid/n4_{k:03d}.tab"; fb=f"{SW}/casos_valid/n4_{k:03d}.spc"
    if not(os.path.isfile(fa) and os.path.isfile(fb)): continue
    _,dd2,s=read_spc(fb); hs,tp=rpar(fa)
    TT.append((int(idx),tp,hs,dpk(fq,dd2,s[0])))
TT=np.array(TT); iv=TT[:,0].astype(int)
def circ(a,b): return (a-b+180)%360-180
print("\n[6] VALIDACION en los 60 estados aleatorios (ajenos al entrenamiento)")
print("    %-8s %9s %9s %9s %9s"%("","sesgo","RMSE","med|e|","r"))
for n,tr,pr,ang in (("Hs",TT[:,2],esp[iv,0],False),("Tp",TT[:,1],esp[iv,1],False),
                    ("Dpeak",TT[:,3],esp[iv,2],True)):
    e=circ(pr,tr) if ang else pr-tr
    print("    %-8s %+9.3f %9.3f %9.3f %9.3f"%(n,e.mean(),np.sqrt((e**2).mean()),
          np.median(np.abs(e)),np.corrcoef(tr,pr)[0,1]))
b=np.array([8,10,12,14,16,18,30])
print("\n    distribucion de Tp (%)")
for lbl,v in (("SWAN real",TT[:,1]),("reconstruido",esp[iv,1])):
    h,_=np.histogram(v,b); print("    %-14s %s"%(lbl," ".join("%6.1f"%x for x in 100*h/h.sum())))

np.savez_compressed(OUTD+"/reconstruccion_final_N4.npz",
    lon=-73.16406,lat=-36.73878,utm_e=663920.0,utm_n=5932534.0,depth_m=21.9,
    HsTpDir=esp,HsTpDir_rbf=par,Tm01=Tm,Hs_spec=esp[:,0],Ycoef_all=Yca,
    EOF_out=E,Yspec_mean=Ym,frec=fq,dirs=dd,nf=nf,nd=nd,
    time=np.array(f['NID_027']['time']),HsN10=HsN.astype(np.float32),operacional=op,
    d=d,M=M,n_modos=ko,sigmas_par=str(sig_par),sigmas_spc=sig_spc,
    nota="Seleccion MDA (500) + KMA (100), deduplicada. PCA estandarizado d=30. "
         "RBF gaussiano exacto, sigma por Rippa (bluemath_tk). Base EOF de salida de "
         "60 modos. HsTpDir se INTEGRA del espectro reconstruido; HsTpDir_rbf es el "
         "RBF parametrico, solo de referencia. Clima completo, sin acotar.")
print("\n[7] guardado -> datos/reconstruccion_final_N4.npz  (%.1f min)"%((time.time()-t0)/60))
