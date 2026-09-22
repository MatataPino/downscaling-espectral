# -*- coding: utf-8 -*-
"""Seleccion de casos por K-means (KMA, density-based) sobre los coeficientes EOF
del clima OPERACIONAL (Hs<3 m) del nodo N10. Sin periodos de retorno -> el foco
es representar el clima medio/operacional (Camus 2011a: KMA para clima medio y
agitacion portuaria). Numero de casos por convergencia (elbow). Devuelve MEDOIDES
(estado de mar real mas cercano a cada centroide) para correr en SWAN."""
import h5py, numpy as np, os, time
from scipy.cluster.vq import kmeans2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A1=r"RAIZ_PROYECTO/Actividad 1"
PCA=r"RAIZ_PROYECTO/Actividad 2/analisis con PCA"
Z=np.load(A1+"/eof_nodo_completo.npz"); mean=Z['mean']; EOF=Z['EOF'][:16]
frec=Z['frec'].ravel(); dirs=Z['dirs'].ravel(); Ndir=int(Z['Ndir']); Nf=int(Z['Nf']); D=Ndir*Nf
f=h5py.File(A1+"/NID_027.mat",'r'); Spec=f['NID_027']['Spec']; Nt=Spec.shape[0]; CH=20000
df=np.gradient(frec); dth=np.deg2rad(np.gradient(dirs))
t0=time.time(); C=np.empty((Nt,16)); Hs=np.empty(Nt); Tp=np.empty(Nt); Dp=np.empty(Nt)
for i in range(0,Nt,CH):
    raw=np.array(Spec[i:i+CH]); C[i:i+CH]=(raw.reshape(-1,D)-mean)@EOF.T
    Hs[i:i+CH]=4*np.sqrt((raw*df[None,None,:]*dth[:,None][None,:,:]).sum((1,2)))
    Sf=raw.sum(1); Tp[i:i+CH]=1/frec[Sf.argmax(1)]; Sd=raw.sum(2); Dp[i:i+CH]=dirs[Sd.argmax(1)]
print("proyeccion %ds"%(time.time()-t0))

# --- banda operacional Hs<3 + estandarizacion (evita que PC1 domine) ---
OPMAX=3.0; op=np.where(Hs<OPMAX)[0]; Cop=C[op]
mu=Cop.mean(0); sd=Cop.std(0); Cs=(Cop-mu)/sd            # z-score -> density-proportional
print("operacional Hs<%.1f: %d estados (%.0f%% del tiempo)"%(OPMAX,len(op),100*len(op)/Nt))
# submuestra para el elbow (rapido)
rng=np.random.default_rng(0); sub=rng.choice(len(Cs),60000,replace=False); Csub=Cs[sub]

Ks=[30,50,75,100,125,150]; inertia=[]
for K in Ks:
    cen,lab=kmeans2(Csub,K,minit='++',seed=0,iter=25,missing='warn')
    inertia.append(float(((Csub-cen[lab])**2).sum(1).mean()))
    print("K=%3d inertia=%.3f t=%ds"%(K,inertia[-1],time.time()-t0))
inertia=np.array(inertia)
x=np.array(Ks,float); xn=(x-x.min())/(x.max()-x.min()); yn=(inertia-inertia.min())/(inertia.max()-inertia.min())
Kelbow=Ks[int(np.argmax((1-xn)-yn))]
Kopt=100   # fijado por el usuario para cubrir el clima operacional
totvar=Cs.var(0).sum()
print("K elbow ~ %d | K elegido = %d"%(Kelbow,Kopt))

# --- K-means final (submuestra) + medoides desde el set operacional completo ---
cen,lab=kmeans2(Csub,Kopt,minit='++',seed=0,iter=50,missing='warn')
expl=1-(((Csub-cen[lab])**2).sum(1).mean())/totvar   # varianza operacional explicada por el clustering
medoids=[]
for c in range(Kopt):
    d=((Cs-cen[c])**2).sum(1); medoids.append(int(op[np.argmin(d)]))
medoids=np.unique(medoids)
print("K=%d captura el %.0f%% de la varianza del clima operacional"%(Kopt,100*expl))
np.savez(PCA+"/datos/kmeans_casos.npz",sel=medoids,C=C[medoids],Hs=Hs[medoids],Tp=Tp[medoids],Dp=Dp[medoids],
         inertia=inertia,Ks=np.array(Ks),Kopt=Kopt,mu=mu,sd=sd)
print("\nK-means operacional: %d medoides | Hs %.2f-%.2f | Tp %.1f-%.1f | Dir %.0f-%.0f"%(
    len(medoids),Hs[medoids].min(),Hs[medoids].max(),Tp[medoids].min(),Tp[medoids].max(),Dp[medoids].min(),Dp[medoids].max()))
for a,b in [(0.8,1.2),(1.2,1.6),(1.6,2.0),(2.0,2.4),(2.4,2.7),(2.7,3.0)]:
    print("  Hs %.1f-%.1f: %d"%(a,b,int(((Hs[medoids]>=a)&(Hs[medoids]<b)).sum())))

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':10,'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in',
 'xtick.minor.visible':True,'ytick.minor.visible':True,'legend.frameon':False,'savefig.dpi':300})
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(Ks,inertia,'o-',color='navy'); ax[0].axvline(Kopt,color='crimson',ls='--',lw=1,label='K = %d (codo)'%Kopt)
ax[0].set_xlabel('Numero de clusters K'); ax[0].set_ylabel('Error de cuantizacion')
ax[0].set_title('(a) Convergencia K-means (elbow)'); ax[0].legend(); ax[0].grid(alpha=.25)
bins=np.arange(0.5,3.6,0.15)
ax[1].hist(Hs[Hs<3.5],bins=bins,density=True,color='0.82',label='Clima operacional N10')
ax[1].hist(Hs[medoids],bins=bins,density=True,histtype='step',color='crimson',lw=1.8,label='%d medoides K-means'%len(medoids))
ax[1].set_xlabel('$H_s$ en el N10 [m]'); ax[1].set_ylabel('Densidad'); ax[1].legend(fontsize=8.5)
ax[1].set_title('(b) K-means sigue la densidad del clima operacional')
plt.tight_layout()
for e in('png','pdf'): fig.savefig(PCA+f"/figuras/paper/Fig_kmeans_seleccion.{e}",bbox_inches='tight')
print("figura -> Fig_kmeans_seleccion")
