# -*- coding: utf-8 -*-
"""Corre los 500 casos estacionarios de la seleccion Camus, en paralelo.

Cada worker trabaja en su propio directorio SWAN/_wK y ejecuta alli swan.exe,
de modo que los ficheros que el modelo escribe en el directorio de trabajo
(PRINT, norm_end, Errfile) no se pisan entre procesos. Las rutas del INPUT se
reescriben con '../' para que malla y salidas sigan siendo las compartidas.

Es REANUDABLE: se saltan los casos cuyas tablas ya existen y tienen datos, asi
que si se interrumpe basta con relanzarlo.

Variables de entorno:
  NW    numero de procesos en paralelo (por defecto 6)
  SOLO  correr solo los primeros N casos (0 = todos), para pruebas

Progreso legible en casos_camus/progress.txt
"""
import os, re, sys, time, shutil, subprocess, queue, threading
from concurrent.futures import ThreadPoolExecutor
WORK=r"RAIZ_PROYECTO\Actividad 2\SWAN"
CAS =os.path.join(WORK,"casos_camus")
SWAN=r"RUTA_SWAN\swan.exe"
NW  =int(os.environ.get("NW","6"))
SOLO=int(os.environ.get("SOLO","0"))

ids=sorted(f.split("_c")[1] for f in os.listdir(CAS) if f.startswith("INPUT_c"))
if SOLO: ids=ids[:SOLO]

def hecho(n):
    for f in (f"n4_{n}.tab",f"adcp_{n}.tab"):
        p=os.path.join(CAS,f)
        if not os.path.isfile(p): return False
        if not any(l.strip() and not l.startswith('%')
                   for l in open(p,encoding='latin-1',errors='replace')): return False
    return True

def leer(n,pre):
    try:
        for ln in open(os.path.join(CAS,f"{pre}_{n}.tab"),encoding='latin-1',errors='replace'):
            if ln.startswith('%') or not ln.strip(): continue
            v=ln.split(); return v[0],v[2],v[3]              # Hs, TPS, Dir
    except Exception: pass
    return "NA","NA","NA"

pend=[n for n in ids if not hecho(n)]
print("[0] %d casos | %d hechos | %d pendientes | %d procesos"
      %(len(ids),len(ids)-len(pend),len(pend),NW))
if not pend: print("nada que hacer"); sys.exit(0)

# Cola de directorios de trabajo. El directorio se toma de la cola al empezar
# un caso y se devuelve al terminar, de modo que NUNCA hay dos procesos SWAN en
# el mismo sitio. (Atarlo al indice del caso, k%NW, es incorrecto: con mas
# casos que workers dos hilos concurrentes acaban compartiendo directorio, se
# pisan el INPUT y se borran el norm_end mutuamente.)
libres=queue.Queue()
for k in range(NW):
    d=os.path.join(WORK,"_w%d"%k); os.makedirs(d,exist_ok=True)
    # SWAN crea swaninit y termina de inmediato la primera vez en un directorio
    # nuevo; se copia de antemano para no perder el primer caso de cada worker.
    ini=os.path.join(WORK,"swaninit")
    if os.path.isfile(ini) and not os.path.isfile(os.path.join(d,"swaninit")):
        shutil.copy2(ini,os.path.join(d,"swaninit"))
    libres.put(d)
lock=threading.Lock()
logp=os.path.join(CAS,"batch_log.csv"); nuevo=not os.path.isfile(logp)
log=open(logp,"a")
if nuevo:
    log.write("caso,status,seg,Hs_adcp,Tp_adcp,Dir_adcp,Hs_n4,Tp_n4,Dir_n4\n"); log.flush()
t0=time.time(); est={"ok":0,"fail":0,"n":0}

def corre(n):
    wd=libres.get()                      # directorio en exclusiva
    try:
        for f in ('PRINT','norm_end','Errfile'):
            p=os.path.join(wd,f)
            if os.path.isfile(p):
                try: os.remove(p)
                except OSError: pass
        txt=open(os.path.join(CAS,f"INPUT_c{n}"),encoding='latin-1').read()
        txt=txt.replace("'Importante/","'../Importante/").replace("'casos_camus/","'../casos_camus/")
        open(os.path.join(wd,"INPUT"),"w",newline="\n").write(txt)
        ts=time.time()
        try:
            subprocess.run([SWAN],cwd=wd,stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=1800)
        except subprocess.TimeoutExpired: pass
        dt=time.time()-ts
        ne=os.path.join(wd,'norm_end')
        good=os.path.isfile(ne) and 'Normal end' in open(ne,encoding='latin-1',errors='replace').read()
    finally:
        libres.put(wd)                   # se devuelve pase lo que pase
    a=leer(n,'adcp'); b=leer(n,'n4')
    st='OK' if (good and hecho(n)) else 'FAIL'
    with lock:
        est["ok"]+=st=='OK'; est["fail"]+=st=='FAIL'; est["n"]+=1
        log.write("%s,%s,%.0f,%s,%s,%s,%s,%s,%s\n"%(n,st,dt,*a,*b)); log.flush()
        el=time.time()-t0; done=est["n"]
        eta=el/done*(len(pend)-done)/60
        with open(os.path.join(CAS,"progress.txt"),"w") as pf:
            pf.write("caso %d/%d (id %s)  OK=%d FAIL=%d  ultimo=%s %.0fs  Hs_N4=%s\n"
                     "transcurrido=%.1f min  ETA=%.1f min  (%d procesos)\n"
                     %(done,len(pend),n,est["ok"],est["fail"],st,dt,b[0],el/60,eta,NW))
    return st

with ThreadPoolExecutor(max_workers=NW) as ex:
    list(ex.map(corre,pend))
log.close()
msg="TERMINADO: %d OK, %d FAIL en %.1f min"%(est["ok"],est["fail"],(time.time()-t0)/60)
with open(os.path.join(CAS,"progress.txt"),"a") as pf: pf.write("\n"+msg+"\n")
print(msg)
