# Acceso a Datasets y Modelos Preentrenados / Dataset & Model Access

> **Aviso para Revisores del Manuscrito (Under Review):**  
> Este repositorio acompaña el artículo *"Real-time operational wave forecasting in ports using Fourier Neural Operators as high-fidelity surrogates"*, actualmente **en revisión por pares** en *Coastal Engineering Journal* (Taylor & Francis).  
> 
> Debido a las restricciones de tamaño de GitHub (>25 GB en total), los archivos de datos voluminosos y los checkpoints de las redes neuronales no se alojan en el repositorio Git. **Cualquier revisor o investigador puede solicitar acceso inmediato a los datasets completos y checkpoints para fines de auditoría académica** contactando directamente a los autores.

---

## Solicitud de Datos / How to Request Data

Para solicitar acceso a cualquiera de los conjuntos de datos o pesos preentrenados, por favor contactar al equipo de investigación del **Instituto Nacional de Hidráulica (INH)**:

- **Contacto principal:** `i-d-i@inh.cl` / `lzamorano@uchile.cl`
- **Institución:** Instituto Nacional de Hidráulica (INH), Chile
- **Asunto sugerido:** `[PortWaveSurrogate] Solicitud de Datasets / Models - Paper San Antonio`

---

## Estructura de Carpetas de Datos / Folder Structure

### 1. Batimetrías (`data/bat/`)
- `5x5_malla_moloBatiMod3.dep`: Batimetría suavizada reciente utilizada en el entrenamiento de los modelos surrogates.
- `5x5_malla_moloBatiMod2.dep`: Batimetría histórica utilizada en la serie de validación 2021.
- *Estado:* Disponible a pedido.

### 2. Hindcast de Oleaje (`data/hindcast/`)
- `FUNW.txt`: Parámetros de oleaje horario extraídos de la simulación espectral SWAN (1980–2013) frente a San Antonio ($H_{m0}$, $T_p$, Dir).
- *Estado:* Disponible a pedido.

### 3. Dataset de Entrenamiento FUNWAVE (`data/funwave_train/`)
- `funwave_maxdiss_dataset.npz` (~4.4 GB): 3000 simulaciones Boussinesq FUNWAVE-TVD seleccionadas mediante el algoritmo Maximum Dissimilarity Algorithm (MDA). Contiene matrices de 425×380 celdas ($dx=5$ m) para $H_{m0}$, $\eta_{\text{mean}}$, $u_{\text{mean}}$, $v_{\text{mean}}$.
- `input_parametros_maxdiss.txt`: Condiciones de borde correspondientes ($H_{m0}$, $1/T_p$, $270 - \text{Dir}$).
- *Estado:* Disponible a pedido.

### 4. Dataset de Evaluación Anual 2021 (`data/funwave_test/`)
- `funwave_2021_dataset.h5` (~20 GB): Serie temporal continua de 8760 horas del año 2021 simulada con FUNWAVE-TVD para validación fuera de distribución (*Out-Of-Distribution*).
- `gt_2021_clean.npz` (~2.2 MB): Serie Ground Truth limpia en los 5 sitios de control portuarios (Sitios A, B, C, D, E).
- *Estado:* Disponible a pedido.

### 5. Checkpoints de Redes Neuronales (`data/checkpoints/`)
- `fno_maxdiss_3000_v2_best.pth` (~133 MB): Pesos óptimos del Fourier Neural Operator (FNO-v2: $k=45$, $d=32$, $\text{pad}=4$, 16.6M parámetros).
- `ulib360_maxdiss_3000_final.keras` (~103 MB): Pesos de la red Attention U-Net a resolución completa de 360×360.
- `tru360_maxdiss_3000_v1_best.keras` (~8.2 GB): Checkpoint del modelo TransUNet.
- *Estado:* Disponible a pedido.

---

## Auditoría sin Datasets Pesados

Los scripts en [`paper_reproduction/`](../paper_reproduction/README_REVIEWERS.md) y la biblioteca [`portsurrogate`](../portsurrogate) están diseñados de forma declarativa: si ya se cuenta con los archivos descargados o en un servidor local, simplemente configure sus rutas en [`configs/san_antonio.yaml`](../configs/san_antonio.yaml).
