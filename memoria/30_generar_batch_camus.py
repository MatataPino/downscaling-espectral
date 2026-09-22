# -*- coding: utf-8 -*-
"""Genera los 500 casos SWAN de la seleccion Camus (PCA estandarizado + MDA).

Para cada caso escribe:
  casos_camus/c{NNN}_w.sp2 / _s.sp2 / _n.sp2   espectro 2D del nodo N10
  casos_camus/INPUT_c{NNN}                     archivo de entrada de SWAN

Los tres .sp2 son identicos: SWAN no admite abrir el mismo fichero en mas de
un contorno, de modo que hay que replicarlo para los lados 2 (O), 3 (S) y 4 (N).

Unidades: el reanalisis entrega la densidad en m2/Hz/rad y con las direcciones
en la primera dimension. SWAN espera m2/Hz/grado y la matriz como
(frecuencia x direccion), de modo que se traspone y se multiplica por pi/180.
Verificado contra la libreria anterior: reproduce su Hs al cuarto decimal.

No lanza nada. Para correr: 31_correr_batch_camus.py
"""
import os, numpy as np, h5py
BASE=r"RAIZ_PROYECTO"
SW  =BASE+"/Actividad 2/SWAN"; PCA=BASE+"/Actividad 2/analisis con PCA"
CAS =SW+"/casos_camus"; TAG="casos_camus"
SEL =PCA+"/datos/camus_std_d30.npz"
FACTOR=1.0e-1
os.makedirs(CAS,exist_ok=True)

K=np.load(SEL); sel=K['sel']; Hs=K['Hs']; Tp=K['Tp']; Dp=K['Dp']
Z=np.load(PCA+"/datos/eof_nodo_completo.npz")
frec=Z['frec'].ravel(); dirs=Z['dirs'].ravel(); nf=len(frec); nd=len(dirs)
f=h5py.File(BASE+"/Actividad 1/NID_027.mat",'r'); Spec=f['NID_027']['Spec']
N=len(sel)
print("[0] %d casos | %d frec x %d dir | destino %s"%(N,nf,nd,TAG))
print("    Hs %.2f-%.2f m | Tp %.1f-%.1f s | Dp %.0f-%.0f deg"
      %(Hs.min(),Hs.max(),Tp.min(),Tp.max(),Dp.min(),Dp.max()))

cab=("SWAN   1                                Swan standard spectral file, version\n"
     "$ Camus: PCA estandarizado + MDA\n"
     "LONLAT                                  locations in spherical coordinates\n"
     "     1                                  number of locations\n"
     "  -76.498  -37.000\n"
     "AFREQ                                   absolute frequencies in Hz\n"
     "    %d                                  number of frequencies\n"%nf
     +"".join("  %.4E\n"%v for v in frec)+
     "NDIR                                    spectral nautical directions in degr\n"
     "    %d                                  number of directions\n"%nd
     +"".join("  %.4f\n"%v for v in dirs)+
     "QUANT\n     1                                  number of quantities in table\n"
     "VaDens                                  variance densities in m2/Hz/degr\n"
     "m2/Hz/degr\n  -0.9900E+02                           exception value\n")

PLANT="""$ SWAN estacionario -- CAMUS (PCA estandarizado + MDA) caso {n} (Hs={hs:.2f} Tp={tp:.1f} Dir={dp:.0f})
PROJECT 'PSV-CAM' 'c{n}'
MODE STATIONARY
COORDINATES SPHERICAL
SET NAUTICAL
SET DEPMIN 0.05
SET MAXERR 3
CGRID UNSTRUCTURED CIRCLE 36 0.03 1.0 32
READGRID UNSTRUCTURED TRIANGLE 'Importante/svicente_mesh'
INPGRID BOTTOM UNSTRUCTURED
READINP BOTTOM 1.0 'Importante/svicente_mesh.bot' FREE
SET EXCMARK 5
BOUNDSPEC SIDE 2 CONSTANT FILE '{t}/c{n}_w.sp2'
BOUNDSPEC SIDE 3 CONSTANT FILE '{t}/c{n}_s.sp2'
BOUNDSPEC SIDE 4 CONSTANT FILE '{t}/c{n}_n.sp2'
GEN3 KOMEN
OFF QUAD
BREAKING
FRICTION JONSWAP 0.038
PROP BSBT
NUMERIC STOPC 0.02 0.02 0.02 98. STAT 50
POINTS 'ADCP' -73.142222 -36.736111
TABLE 'ADCP' HEAD '{t}/adcp_{n}.tab' HS RTP TPS DIR DSPR TM01 TM02
SPECOUT 'ADCP' SPEC2D ABS '{t}/adcp_{n}.spc'
POINTS 'N4' -73.16406 -36.73878
TABLE 'N4' HEAD '{t}/n4_{n}.tab' HS RTP TPS DIR DSPR TM01 TM02
SPECOUT 'N4' SPEC2D ABS '{t}/n4_{n}.spc'
POINTS 'BORDE' -73.17212 -36.75134 -73.16088 -36.74058 -73.14963 -36.72982
TABLE 'BORDE' HEAD '{t}/borde_{n}.tab' XP YP HS RTP TPS DIR DSPR TM01 TM02
SPECOUT 'BORDE' SPEC2D ABS '{t}/borde_{n}.spc'
COMPUTE STATIONARY
STOP
"""

df=np.gradient(frec); ddeg=np.abs(np.gradient(dirs))
chk=[]
for k in range(N):
    n="%03d"%k
    raw=np.array(Spec[int(sel[k])])                 # (ndir, nf), m2/Hz/rad
    M=raw.T*(np.pi/180.0)                           # (nf, ndir), m2/Hz/deg
    cuerpo="FACTOR\n  %.4E\n"%FACTOR+"".join(
        "  ".join("%.4E"%v for v in (M[r]/FACTOR))+"\n" for r in range(nf))
    txt=cab+cuerpo
    for suf in ("w","s","n"):
        open(f"{CAS}/c{n}_{suf}.sp2","w",newline="\n").write(txt)
    open(f"{CAS}/INPUT_c{n}","w",newline="\n").write(
        PLANT.format(n=n,t=TAG,hs=Hs[k],tp=Tp[k],dp=Dp[k]))
    chk.append(4*np.sqrt((M*df[:,None]*ddeg[None,:]).sum()))
    if (k+1)%100==0: print("    %d/%d"%(k+1,N))

chk=np.array(chk)
d=np.abs(chk-Hs)
print("[1] verificacion Hs escrito vs seleccionado: max dif %.2e m  (media %.2e)"%(d.max(),d.mean()))
np.save(CAS+"/meta.npy",np.c_[sel,Hs,Tp,Dp])
print("[2] listo: %d INPUT + %d sp2 en %s"%(N,3*N,TAG))
print("    meta.npy -> columnas [indice_horario, Hs, Tp, Dpeak] del nodo N10")
