# Downscaling espectral híbrido de oleaje

Metodología para transferir un registro largo de espectros direccionales de
oleaje desde un nodo oceánico hasta un punto costero, propagando con SWAN solo
una fracción de los estados de mar y reconstruyendo el resto por interpolación.

Desarrollada en la memoria de título *[título de la memoria]*, Departamento de
Ingeniería Civil, Universidad de Chile, y aplicada a la bahía de San Vicente:
46,5 años de espectros horarios (407.592 estados) reconstruidos a partir de 659
propagaciones, el 0,16 % del registro.

Sigue la metodología de Camus et al. (2011a, 2011b, 2013), con una diferencia:
tanto la entrada como la salida del interpolador son **espectros direccionales
completos**, no parámetros integrados. Los parámetros de estado de mar se
obtienen integrando el espectro reconstruido, de modo que espectro y parámetros
describen el mismo estado de mar.

## Estructura

```
config/          un archivo por sitio: todo lo específico del lugar
downscaling/     el paquete: lectura, PCA, selección, SWAN, RBF, reconstrucción
pasos/           los seis pasos del pipeline, en orden
tests/           prueba de que el pipeline reproduce la memoria
memoria/         los programas tal como se ejecutaron en la memoria
```

## El método en seis pasos

| Paso | Programa | Qué hace | Escribe |
|---|---|---|---|
| 1 | `p1_base_eof.py` | base EOF de los espectros centrados del nodo | `base_eof_nodo.npz` |
| 2 | `p2_pca_mda.py` | PCA de los espectros estandarizados; selección por máxima disimilitud | `seleccion_mda.npz` |
| 3 | `p3_kmedias.py` | K-medias sobre el clima operacional; estados de verificación | `seleccion_kmedias.npz`, `verificacion.npy` |
| 4 | `p4_generar_swan.py` | espectros de contorno y archivos de entrada de SWAN | carpetas de casos |
| 5 | `p5_correr_swan.py` | propagación estacionaria, en paralelo y reanudable | espectros en el punto |
| 6 | `p6_reconstruccion.py` | 64 interpoladores RBF y reconstrucción del registro | `reconstruccion_<punto>.npz` |

La máxima disimilitud cubre la frontera del espacio de estados —ningún
temporal queda fuera del alcance del interpolador— y K-medias cubre la
densidad del régimen frecuente, que la primera ignora por construcción. Es la
complementariedad que establecen Camus et al. (2011a).

## Instalación

Python 3.12 y SWAN 41.51.

```
pip install -r requirements.txt
```

## Uso

Definir dónde están los datos y el ejecutable de SWAN:

```
set MEMORIA=C:\ruta\a\los\datos
set SWAN_EXE=C:\ruta\a\swan.exe
```

y ejecutar los pasos en orden desde la raíz del repositorio:

```
python pasos/p1_base_eof.py
python pasos/p2_pca_mda.py
python pasos/p3_kmedias.py
python pasos/p4_generar_swan.py
python pasos/p5_correr_swan.py
python pasos/p6_reconstruccion.py
```

Todos aceptan `--config` para usar otro sitio; por defecto emplean
`config/san_vicente.toml`.

## Aplicarlo a otro sitio

Copiar `config/san_vicente.toml` y cambiar sus valores:

| Sección | Qué contiene |
|---|---|
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
espectral y esa información vive en los modos de orden alto. En San Vicente,
21 modos —el 99,8 % de la varianza— daban 14,6 s de error en el período; con
60 baja a 0,19 s.

El registro del nodo debe ser un archivo HDF5 (`.mat` v7.3) con un grupo que
contenga `Spec` (Nt × Ndir × Nf, en m²/Hz/rad), `frec`, `dir` y `time`. Otro
formato requiere adaptar `downscaling/espectros.py`.

## Reproducción de la memoria

```
python tests/reproducir_san_vicente.py
```

Ejecuta los pasos 1 a 4 y 6 con la configuración de San Vicente —el paso 5 no
se repite: el 6 usa las propagaciones existentes— y compara cada salida con el
archivo empleado en la memoria.

| Paso | Resultado |
|---|---|
| 1. Base EOF | idéntica bit a bit |
| 2. PCA + máxima disimilitud | idéntica bit a bit: los mismos 500 casos, en el mismo orden |
| 3. K-medias | idéntica bit a bit |
| 4. Entradas de SWAN | idénticas, salvo el factor de escala de 11 contornos de K-medias |
| 6. Reconstrucción | coincide con las cifras publicadas a la precisión publicada |

### Notas de reproducibilidad

**Paso 6.** SWAN no reproduce sus salidas bit a bit entre corridas, de modo que
la reconstrucción se entrena sobre espectros que difieren en milímetros de los
que se usaron en la memoria: 0,5 mm en $H_s$ como mediana sobre las 407.592
horas y 8,6 mm como máximo. No altera ninguna cifra publicada salvo el RMSE del
período de pico en la verificación, que pasa de 2,401 a 2,402 s.

**Contornos de K-medias.** El generador con que se escribieron ajustaba el
factor de escala caso a caso: 11 de los 100 usan `FACTOR` = 10⁻² en vez de
10⁻¹. El formato `.sp2` guarda enteros escalados por ese factor, de modo que la
densidad espectral que lee SWAN es la misma; la diferencia queda en el redondeo
del último dígito impreso (relativa, menor que 10⁻⁴).

**Estados de verificación.** Los 60 estados de verificación se fijan como lista
en `config/san_vicente_verificacion.txt`. Para otro sitio, si se omite la
lista, el paso 3 los sortea con la semilla de la configuración.

## Verificación de la implementación

El interpolador sigue la formulación de Camus et al. (2011b): núcleo gaussiano,
base monomial de grado 1, ajuste exacto y parámetro de forma por validación
cruzada de Rippa (1999). Se contrastó con la biblioteca BlueMath_tk
(IH Cantabria) mediante `memoria/35_verificacion_bluemath.py`: la selección por
máxima disimilitud coincide en los 500 casos y las predicciones difieren en
menos de 10⁻¹¹ para un mismo parámetro de forma. Se usa una implementación
propia porque comparte la factorización del sistema entre los 64
interpoladores, lo que reduce el tiempo de ajuste de horas a segundos.

## Uso de inteligencia artificial

El código de este repositorio se desarrolló con asistencia de una herramienta
de inteligencia artificial generativa (Claude, Anthropic). Los resultados se
verificaron contra la biblioteca BlueMath_tk, contra propagaciones directas con
SWAN y contra mediciones de un perfilador acústico (ADCP).

## Referencias

- Camus, P., Mendez, F. J., Medina, R. y Cofiño, A. S. (2011a). Analysis of clustering and selection algorithms for the study of multivariate wave climate. *Coastal Engineering*, 58, 453–462.
- Camus, P., Mendez, F. J. y Medina, R. (2011b). A hybrid efficient method to downscale wave climate to coastal areas. *Coastal Engineering*, 58, 851–862.
- Camus, P., Mendez, F. J., Medina, R., Tomas, A. e Izaguirre, C. (2013). High resolution downscaled ocean waves (DOW) reanalysis in coastal areas. *Coastal Engineering*, 72, 56–68.
- Kennard, R. W. y Stone, L. A. (1969). Computer aided design of experiments. *Technometrics*, 11, 137–148.
- Rippa, S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193–210.
