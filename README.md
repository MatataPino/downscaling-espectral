# Downscaling espectral híbrido de oleaje

Transfiere un registro largo de espectros direccionales desde un nodo oceánico
hasta un punto costero propagando con SWAN solo una fracción de los estados de
mar y reconstruyendo el resto por interpolación. Con unos cientos de
propagaciones se obtiene el registro completo: décadas de espectros horarios en
el punto de interés, por un tiempo de cálculo de horas en vez de meses.

Sigue la metodología de Camus et al. (2011a, 2011b, 2013), con una diferencia:
tanto la entrada como la salida del interpolador son **espectros direccionales
completos**, no parámetros integrados. Los parámetros de estado de mar se
obtienen integrando el espectro reconstruido, de modo que espectro y parámetros
describen el mismo estado de mar.

## El método en seis pasos

| Paso | Programa | Qué hace | Escribe |
|---|---|---|---|
| 1 | `p1_base_eof.py` | base EOF de los espectros centrados del nodo | `base_eof_nodo.npz` |
| 2 | `p2_pca_mda.py` | PCA de los espectros estandarizados; selección por máxima disimilitud | `seleccion_mda.npz` |
| 3 | `p3_kmedias.py` | K-medias sobre el clima operacional; estados de verificación | `seleccion_kmedias.npz`, `verificacion.npy` |
| 4 | `p4_generar_swan.py` | espectros de contorno y archivos de entrada de SWAN | carpetas de casos |
| 5 | `p5_correr_swan.py` | propagación estacionaria, en paralelo y reanudable | espectros en el punto |
| 6 | `p6_reconstruccion.py` | interpoladores RBF y reconstrucción del registro | `reconstruccion_<punto>.npz` |

La máxima disimilitud cubre la frontera del espacio de estados —ningún temporal
queda fuera del alcance del interpolador— y K-medias cubre la densidad del
régimen frecuente, que la primera ignora por construcción. Es la
complementariedad que establecen Camus et al. (2011a).

La interpolación sigue la formulación de Camus et al. (2011b): núcleo
gaussiano, base monomial de grado 1, ajuste exacto y parámetro de forma elegido
por validación cruzada de Rippa (1999). Se ajusta un interpolador por cada
parámetro de estado de mar y uno por cada modo de la base de salida, todos
sobre las mismas componentes principales de entrada, de modo que comparten la
factorización del sistema: ajustarlos cuesta segundos.

## Estructura

```
config/          un archivo por sitio: todo lo específico del lugar
downscaling/     el paquete: lectura, PCA, selección, SWAN, RBF, reconstrucción
pasos/           los seis pasos del pipeline, en orden
```

## Instalación

Python 3.12 y SWAN 41.51.

```
pip install -r requirements.txt
```

## Uso

Copiar `config/ejemplo.toml`, ajustar sus valores al sitio, y ejecutar los
pasos en orden desde la raíz del repositorio:

```
python pasos/p1_base_eof.py --config config/mi_sitio.toml
python pasos/p2_pca_mda.py --config config/mi_sitio.toml
python pasos/p3_kmedias.py --config config/mi_sitio.toml
python pasos/p4_generar_swan.py --config config/mi_sitio.toml
python pasos/p5_correr_swan.py --config config/mi_sitio.toml
python pasos/p6_reconstruccion.py --config config/mi_sitio.toml
```

El paso 5 es el largo —es SWAN— y es reanudable: si se interrumpe, basta con
relanzarlo. Los demás son minutos.

Las rutas de la configuración admiten variables de entorno, de modo que el
archivo no queda atado a un computador:

```
set DATOS=C:\ruta\a\los\datos
set SWAN_EXE=C:\ruta\a\swan.exe
```

## Configuración

| Sección | Qué contiene |
|---|---|
| `[rutas]` | dónde están los datos y dónde se escriben los resultados |
| `[nodo]` | archivo del registro espectral y coordenadas del nodo |
| `[pca]` | número de componentes $d$ |
| `[seleccion]` | casos de máxima disimilitud y K-medias, umbral del clima operacional |
| `[swan]` | ejecutable, malla, física, lados de contorno y **puntos de salida** |
| `[rbf]` | punto de destino, modos de la base de salida, malla de σ |

Dos decisiones que dependen del sitio y conviene revisar:

**El número de componentes $d$.** No se fija por un umbral de varianza sino por
el error de reconstrucción, que el paso 2 imprime para varios valores de $d$. Se
elige el menor que alcance un error aceptable en $H_s$, período y dirección.

**Los modos de la base de salida.** Con pocos modos el período de pico se
degrada mucho antes que la altura, porque depende de la posición del máximo
espectral y esa información vive en los modos de orden alto. En una aplicación
a una bahía abierta del Pacífico sur, 21 modos —el 99,8 % de la varianza— daban
14,6 s de error en el período; con 60 baja a 0,19 s.

El paso 3 aparta además un conjunto de estados ajenos al entrenamiento, que se
propagan con los demás, y el paso 6 informa el error del interpolador sobre
ellos. Es la forma de comprobar que las dos decisiones anteriores son
adecuadas para el sitio.

## Formato de los datos

El registro del nodo debe ser un archivo HDF5 (`.mat` v7.3) con un grupo que
contenga `Spec` (Nt × Ndir × Nf, en m²/Hz/rad), `frec`, `dir` y `time`. Otro
formato requiere adaptar `downscaling/espectros.py`.

La malla de SWAN es una malla no estructurada de TRIANGLE (`.node`, `.ele`), y
los lados del contorno donde se impone el espectro del nodo se indican por sus
marcadores.

## Verificación de la implementación

El interpolador se contrastó con la biblioteca BlueMath_tk (IH Cantabria): la
selección por máxima disimilitud coincide caso a caso y las predicciones
difieren en menos de 10⁻¹¹ para un mismo parámetro de forma. Los resultados del
método completo se contrastaron además con propagaciones directas con SWAN y
con mediciones de un perfilador acústico (ADCP).

## Uso de inteligencia artificial

El código de este repositorio se desarrolló con asistencia de una herramienta
de inteligencia artificial generativa (Claude, Anthropic), y fue revisado y
verificado por el autor.

## Referencias

- Camus, P., Mendez, F. J., Medina, R. y Cofiño, A. S. (2011a). Analysis of clustering and selection algorithms for the study of multivariate wave climate. *Coastal Engineering*, 58, 453–462.
- Camus, P., Mendez, F. J. y Medina, R. (2011b). A hybrid efficient method to downscale wave climate to coastal areas. *Coastal Engineering*, 58, 851–862.
- Camus, P., Mendez, F. J., Medina, R., Tomas, A. e Izaguirre, C. (2013). High resolution downscaled ocean waves (DOW) reanalysis in coastal areas. *Coastal Engineering*, 72, 56–68.
- Kennard, R. W. y Stone, L. A. (1969). Computer aided design of experiments. *Technometrics*, 11, 137–148.
- Rippa, S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193–210.
