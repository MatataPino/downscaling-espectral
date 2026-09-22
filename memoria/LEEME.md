# Los programas tal como se ejecutaron

Esta carpeta conserva los programas con que se obtuvieron los resultados de la
memoria, sin reorganizar: cada uno es un archivo suelto, con las rutas escritas
dentro y la numeración con que se fueron escribiendo. Se guardan como
referencia y para poder auditar el pipeline contra ellos; para aplicar el
método conviene usar `pasos/`, que hace lo mismo leyendo la configuración.

| Programa | Qué hace | Dónde vive ahora |
|---|---|---|
| `01_eof_nodo_completo.py` | base EOF de los espectros centrados | `downscaling/pca.py` → `base_centrada` |
| `29_camus_estandarizado.py` | PCA de los espectros estandarizados y máxima disimilitud | `downscaling/pca.py` → `pca_estandarizado`, `downscaling/seleccion.py` → `mda` |
| `08_kmeans_seleccion.py` | K-medias sobre el clima operacional | `downscaling/seleccion.py` → `kmedias` |
| `30_generar_batch_camus.py` | escritura de contornos `.sp2` y archivos de entrada | `downscaling/swan.py` → `escribir_casos` |
| `31_correr_batch_camus.py` | propagación en paralelo, reanudable | `downscaling/swan.py` → `correr` |
| `39_reconstruccion_MDA_KMA.py` | interpoladores RBF y reconstrucción | `downscaling/rbf.py`, `downscaling/reconstruccion.py` |
| `35_verificacion_bluemath.py` | contraste con BlueMath_tk | no tiene equivalente: es una verificación puntual |

Las rutas absolutas del computador en que se ejecutaron se reemplazaron por las
constantes `RAIZ_PROYECTO`, `RUTA_SWAN` y `RUTA_DATOS` al comienzo de cada
archivo. Fuera de eso el código es el que se ejecutó.

`tests/reproducir_san_vicente.py` comprueba que el pipeline reproduce lo que
produjeron estos programas. Los pasos 1 a 3 coinciden bit a bit; el detalle de
las diferencias que quedan está en el README.
