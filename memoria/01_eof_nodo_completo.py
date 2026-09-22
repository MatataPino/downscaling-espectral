# -*- coding: utf-8 -*-
import h5py, numpy as np, time
t0=time.time()
f = h5py.File('NID_027.mat','r'); g=f['NID_027']
Spec = g['Spec']                      # (407592,24,29) en disco
dirs = np.array(g['dir']).ravel(); frec=np.array(g['frec']).ravel()
Nt, Ndir, Nf = Spec.shape; D = Ndir*Nf
print('Nt=%d  dim=%d' % (Nt, D))

# --- paso 1: media por bloques ---
CH = 20000
ssum = np.zeros(D)
for i in range(0, Nt, CH):
    blk = np.array(Spec[i:i+CH]).reshape(-1, D)
    ssum += blk.sum(0)
mean = ssum / Nt
print('media lista  t=%.0fs' % (time.time()-t0))

# --- paso 2: covarianza 696x696 por bloques ---
Cov = np.zeros((D, D))
for i in range(0, Nt, CH):
    blk = np.array(Spec[i:i+CH]).reshape(-1, D) - mean
    Cov += blk.T @ blk
Cov /= (Nt - 1)
print('covarianza lista  t=%.0fs' % (time.time()-t0))

# --- paso 3: autovectores = EOFs (componentes principales) ---
w, V = np.linalg.eigh(Cov)          # ascendente
idx = np.argsort(w)[::-1]
w = w[idx]; V = V[:, idx]
vr = w / w.sum(); cum = np.cumsum(vr)
n95 = int(np.searchsorted(cum, 0.95)+1)
n99 = int(np.searchsorted(cum, 0.99)+1)

print('\n=== COMPONENTES PRINCIPALES (nodo completo, 46.5 anios) ===')
for k in range(9):
    print('  PC%-2d : %6.2f %%   (acumulado %6.2f %%)' % (k+1, 100*vr[k], 100*cum[k]))
print('  ...')
print('  9 EOFs -> %.2f%% | 95%%: %d EOFs | 99%%: %d EOFs' % (100*cum[8], n95, n99))

# guardar los 9 componentes + media + varianzas para el pipeline
np.savez('eof_nodo_completo.npz',
         mean=mean, EOF=V[:,:40].T, var_ratio=vr[:40],
         frec=frec, dirs=dirs, Ndir=Ndir, Nf=Nf, Nt=Nt)
print('\nGuardado -> eof_nodo_completo.npz (15 primeros modos)  t=%.0fs' % (time.time()-t0))
