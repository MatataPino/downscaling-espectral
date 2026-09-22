# Downscaling espectral híbrido — Bahía de San Vicente

Códigos de la metodología de *downscaling* espectral desarrollada en la memoria
de título *[título de la memoria]*, Departamento de Ingeniería Civil,
Universidad de Chile.

El método transfiere 46,5 años de espectros direccionales horarios del nodo
oceánico NID_027 (1979–2025) a la bahía de San Vicente propagando con SWAN solo
659 estados de mar —el 0,16 % del registro— y reconstruyendo el resto mediante
funciones de base radial. Sigue la metodología de Camus et al. (2011a, 2011b,
2013), con una diferencia: tanto la entrada como la salida del interpolador son
**espectros direccionales completos**, no parámetros integrados.

## Programas, en orden de ejecución

| | Programa | Qué hace | Produce |
|---|---|---|---|
| 1 | `29_camus_estandarizado.py` | PCA sobre los 696 valores espectrales estandarizados; selección por máxima disimilitud | 500 casos |
| 2 | `08_kmeans_seleccion.py` | K-medias sobre el clima operacional ($H_s$ < 3 m); medoides | 100 casos |
| 3 | `30_generar_batch_camus.py` | espectros de contorno y archivos de entrada de SWAN | entradas |
| 4 | `31_correr_batch_camus.py` | propagación estacionaria en paralelo | espectros en N4 |
| 5 | `39_reconstruccion_MDA_KMA.py` | base EOF de salida, 64 interpoladores RBF, reconstrucción | 407.592 espectros |
| — | `35_verificacion_bluemath.py` | contraste con la biblioteca BlueMath_tk | verificación |

Los programas no se importan entre sí: se comunican a través de ficheros de
datos intermedios (`.npz`), que no se incluyen en el repositorio.

## Datos necesarios (no incluidos)

| Dato | Descripción |
|---|---|
| `Actividad 1/NID_027.mat` | espectros direccionales horarios del nodo oceánico, 29 frecuencias × 24 direcciones, 1979–2025 |
| `Actividad 1/eof_nodo_completo.npz` | base EOF del nodo, empleada por la selección por K-medias |
| `Actividad 2/SWAN/Importante/svicente_mesh.*` | malla no estructurada de SWAN (`.node`, `.ele`, `.bot`) |
| `Programa SWAN/AP_San_Vicente.mat` | espectros del nodo en el periodo instrumental, agosto–septiembre 2025 |
| registro del ADCP (`.csv`) | solo para el programa de verificación |

## Configuración

Las rutas absolutas del equipo original se sustituyeron por tres marcadores,
definidos al inicio de cada programa:

- `RAIZ_PROYECTO` — carpeta raíz del proyecto
- `RUTA_SWAN` — carpeta que contiene `swan.exe`
- `RUTA_DATOS` — carpeta con el registro del ADCP

Hay que reemplazarlos por las rutas locales antes de ejecutar.

## Requisitos

- Python 3.12 y las bibliotecas de `requirements.txt`
- SWAN 41.51

```
pip install -r requirements.txt
```

## Verificación

La implementación del interpolador sigue la formulación de Camus et al. (2011b)
—núcleo gaussiano, base monomial de grado 1, ajuste exacto y parámetro de forma
por validación cruzada de Rippa (1999)— y se contrastó contra BlueMath_tk
(IH Cantabria): la selección por máxima disimilitud coincide en los 500 casos y
las predicciones difieren en menos de 1e-11 para un mismo parámetro de forma.
Se usa una implementación propia porque comparte la factorización del sistema
entre los 64 interpoladores, lo que reduce el tiempo de ajuste de horas a
segundos.

## Asistencia en la programación

La implementación se desarrolló con asistencia de una herramienta de
inteligencia artificial (Claude, Anthropic). Los resultados se verificaron
contra BlueMath_tk y contra propagaciones directas con SWAN.

## Referencias

- Camus, P., Mendez, F. J., Medina, R. y Cofiño, A. S. (2011a). Analysis of clustering and selection algorithms for the study of multivariate wave climate. *Coastal Engineering*, 58, 453–462.
- Camus, P., Mendez, F. J. y Medina, R. (2011b). A hybrid efficient method to downscale wave climate to coastal areas. *Coastal Engineering*, 58, 851–862.
- Camus, P., Mendez, F. J., Medina, R., Tomas, A. e Izaguirre, C. (2013). High resolution downscaled ocean waves (DOW) reanalysis in coastal areas. *Coastal Engineering*, 72, 56–68.
- Rippa, S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193–210.
