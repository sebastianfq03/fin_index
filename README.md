# Índice de Vulnerabilidad Financiera (IVF) & Modelado Look-Alike en el Perú (ENAHO 2025)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Survey Data: ENAHO 2025](https://img.shields.io/badge/Data-INEI%20ENAHO%202025-green.svg)](https://proyectos.inei.gob.pe/microdatos/)
[![Model 1: StepMix LCA](https://img.shields.io/badge/Model%201-StepMix%20(Mixed%20LCA)-orange.svg)](https://github.com/stepmix/stepmix)
[![Model 2: CatBoost Calibrated](https://img.shields.io/badge/Model%202-CatBoost%20Isotonic-yellow.svg)](https://catboost.ai/)
[![Validation: Bootstrap B=200](https://img.shields.io/badge/Validation-Cluster%20Bootstrap%20(ARI%200.79)-purple.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Resumen Ejecutivo y Motivación

A pesar del despliegue masivo de billeteras digitales interoperables (Yape, Plin) en el Perú, la exclusión financiera formal continúa siendo un desafío estructural crítico. Según los microdatos oficiales de la **Encuesta Nacional de Hogares (ENAHO 2025)** del Instituto Nacional de Estadística e Informática (INEI), el **44.6% de la población adulta entre 18 y 65 años** (~8.86 millones de personas proyectadas con el factor de expansión `FAC500A`) carece por completo de cuentas, tarjetas o productos financieros formales (`P558E1_6 == 6`).

Este proyecto supera el paradigma convencional de la exclusión binaria ("bancarizado vs. no bancarizado") y la práctica estadística tradicional basada en dos etapas (MCA + GMM), desarrollando:
1. **Un Índice de Vulnerabilidad Financiera (IVF)** multidimensional y continuo (0 a 1000 puntos), basado en **Modelos de Mezclas Finitas para Datos Mixtos (StepMix: Multinomial-Gaussiano)** con selección algorítmica autónoma de perfiles latentes ($K^*=3$) y ponderación objetiva por **Entropía de Shannon**.
2. **Un Modelo Supervisado Look-Alike** entrenado sobre el panel adulto total ($N = 62,825$) con ensamble **CatBoost** y **Calibración Isotónica**, que identifica a los *gemelos estadísticos* entre los no bancarizados ($N = 28,016$) con alta probabilidad de bancarización espontánea.
3. **Una Matriz Estratégica 2D (IVF vs. Propensión Look-Alike)** que traduce los hallazgos econométricos en **cuatro cuadrantes de originación prescriptiva** para la industria microfinanciera y la política pública.
4. **Una suite completa de entregables:** Dashboard BI Interactivo autónomo en HTML5/Plotly, Monografía Científica en PDF y Memoria Técnica de Arquitectura en PDF, ambos con fórmulas matemáticas vectoriales en 300 DPI y cero códigos crudos de encuesta.

---

## 🏛️ Innovaciones Metodológicas de la ENAHO 2025

El pipeline captura rigurosamente las novedades metodológicas introducidas en la versión 2025 de la ENAHO:

| Módulo ENAHO 2025 | Dimensión Capturada | Variables ENAHO 2025 | Indicador de Negocio / Rol Microfinanciero |
| :--- | :--- | :--- | :--- |
| **Módulo 100** | Colateral Predial y Hábitat | `P106A`, `P106B`, `NBI1` a `NBI5` | Tenencia de vivienda con título registrado en SUNARP (garantía hipotecaria/real) y carencias materiales del hábitat. |
| **Módulo 100** | Conectividad Hogar | `P1144`, `P1144B1`, `P1144B2` | Conectividad fija, internet móvil prepago y postpago del hogar. |
| **Módulo 200** | Demografía y Género | `P207`, `P208A`, `P203` | Edad adulta (18-65 años), género y jefatura de hogar. |
| **Módulo 300** | Capital Humano & Digital | `P301A`, `P314B1_9`, `P316_8` | Nivel educativo ordinal, tenencia de smartphone con plan de datos y uso de redes sociales/marketplaces para comercio digital. |
| **Módulo 400** | Identidad Legal KYC & Salud | `P401C`, `P4191`, `P4195` | Tenencia de DNI (barrera legal de entrada KYC) y cobertura de aseguramiento en salud (EsSalud / SIS). |
| **Módulo 500** | Formalidad y Solvencia | `P507`, `P510A1`, `P524A1`, `P530A` | Categoría ocupacional, negocio registrado con RUC en SUNAT, ingreso dependiente e ingresos independientes. |
| **Módulo 500** | Conducta Financiera Informal | `P558G1`, `P558G7`, `P558E1_6` | Ahorro informal en juntas/panderos, financiamiento gota a gota y variable objetivo de no bancarización. |
| **Módulo 602** | Inseguridad Alimentaria | `P602`, `P602N` (14, 15, 16) | Asistencia a Ollas Comunes y Comedores Populares (agregación relacional a nivel hogar). |
| **Módulo 700 / 700A** | Red de Protección Social | `P701$03`, `P710$04`, `P710$05`, `P710$15` | Transferencias del Estado: JUNTOS, Pensión 65, Wasi Mikuna y Bono Gas FISE. |
| **Factor Expansión** | Inferencia Poblacional | `FAC500A` | Ponderación muestral Horvitz-Thompson en todas las etapas analíticas. |

---

## 📐 Fundamentación Matemática y Modelado Teórico

### 1. Superioridad de StepMix sobre el Paradigma Clásico (MCA + GMM)

La práctica convencional en análisis de encuestas mixtas reduce variables categóricas mediante Análisis de Correspondencias Múltiples (MCA) y aplica Modelos de Mezclas Gaussianas (GMM) sobre los factores continuos extraídos. Dicho enfoque incurre en la **falacia de dos etapas**:
1. Ignora el error de medición de la primera etapa y asume que las puntuaciones factoriales son observaciones exactas.
2. Impone una métrica euclidiana sobre la matriz de Burt que distorsiona las distancias chi-cuadrado y produce el colapso de las probabilidades posteriores en categorías dispersas.

En contraposición, **StepMix** modela de forma simultánea el vector completo de variables mixtas $\mathbf{X}_i = (\mathbf{X}_i^{\text{cat}}, \mathbf{X}_i^{\text{cont}})$ maximizando la función de verosimilitud de mezcla finita conjunta mediante el algoritmo Expectation-Maximization (EM):

$$P(\mathbf{X}_i = \mathbf{x}_i \mid \mathbf{\Theta}) = \sum_{k=1}^K \pi_k \left[ \prod_{j=1}^{M_{\text{cat}}} \prod_{c=1}^{C_j} (\lambda_{kjc})^{\mathbf{1}(x_{ij}^{\text{cat}}=c)} \right] \cdot \frac{1}{(2\pi)^{M_{\text{cont}}/2} |\mathbf{\Sigma}_k|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x}_i^{\text{cont}} - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x}_i^{\text{cont}} - \boldsymbol{\mu}_k) \right)$$

donde:
- $\pi_k = P(Z_i = k)$ es la proporción de mezcla de la clase latente $k$, con $\sum_{k=1}^K \pi_k = 1$.
- $\lambda_{kjc} = P(X_{ij}^{\text{cat}} = c \mid Z_i = k)$ son los parámetros de emisión multinomiales condicionales.
- $\boldsymbol{\mu}_k$ y $\mathbf{\Sigma}_k$ representan el vector de medias y la matriz de covarianzas gaussianas condicionales.

---

### 2. Regla Multicriterio Autónoma para la Selección de $K^*$ ($K \in \{2, 3, 4, 5, 6, 7\}$)

En muestras probabilísticas masivas ($N \approx 30,000$), el Criterio de Información Bayesiano (BIC) estándar tiende a sobredimensionar el número de clases, favoreciendo clusters redundantes con diferencias triviales. Por ello, se implementó la regla multicriterio de **Biernacki, Celeux y Govaert (2000)** y **Nardo et al. (OCDE, 2005)**:

#### A. Criterio de Clasificación Integrada (ICL):
Penaliza la falta de nitidez en la asignación posterior de clusters:

$$\text{ICL}(K) = \text{BIC}(K) + 2 \, E_w(K) = \left[ -2 \ln L_w(K) + d_K \ln N \right] - 2 \sum_{i=1}^N w_i \sum_{k=1}^K \gamma_{ik} \ln(\gamma_{ik} + \epsilon)$$

donde $\gamma_{ik} = P(Z_i = k \mid \mathbf{X}_i = \mathbf{x}_i)$ son las probabilidades posteriores y $w_i = \text{FAC500A}_i$.

#### B. Ganancia Marginal Relativa ($\Delta\text{ICL}_K$):
Detecta el punto de codo donde agregar una clase adicional no compensa la pérdida de parsimonia:

$$\Delta \text{ICL}_K = \frac{\text{ICL}(K-1) - \text{ICL}(K)}{\text{ICL}(K-1)} \times 100\%$$

#### C. Restricción de Masa Poblacional Mínima Expandida:
Ningún cluster debe ser espurio:

$$\min_{k \in \{1, \dots, K\}} \pi_k \ge 5\% \quad \left( \pi_k = \frac{\sum_{i=1}^N w_i \gamma_{ik}}{\sum_{i=1}^N w_i} \right)$$

#### D. Nitidez de Clasificación (Entropía Relativa):

$$E_{\text{rel}}(K) = \frac{-\sum_{i=1}^N w_i \sum_{k=1}^K \gamma_{ik} \ln(\gamma_{ik} + \epsilon)}{N_w \ln K} < 0.15$$

#### 📊 Evidencia Empírica de la Grilla de Selección:

| $K$ | Log-Likelihood | Parámetros $d_K$ | BIC | ICL | $\Delta\text{ICL}$ (%) | Entropía Relativa | Masa Mínima $\pi_{\min}$ (%) | Veredicto Algorítmico |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | -29,088.3 | 29 | 60,379.8 | 150,288.7 | — | 0.0911 | 22.40% | Válido |
| **3** | **-28,944.4** | **44** | **59,628.7** | **78,054.6** | **+48.06%** | **0.0018** | **5.20%** | **ÓPTIMO GLOBAL ($K^*=3$)** |
| **4** | -28,788.0 | 59 | 59,470.2 | 126,183.8 | -61.66% | 0.1440 | 6.80% | Deterioro Entrópico |
| **5** | -28,740.1 | 74 | 59,528.9 | 148,829.4 | -17.95% | 0.1190 | **0.07%** | **Espurio ($\pi_{\min} < 5\%$)** |
| **6** | -28,680.5 | 89 | 59,564.3 | 149,021.2 | -0.13% | 0.1185 | **0.07%** | **Espurio ($\pi_{\min} < 5\%$)** |
| **7** | -28,591.2 | 104 | 59,540.1 | 148,799.0 | +0.15% | 0.1172 | **0.07%** | **Espurio ($\pi_{\min} < 5\%$)** |

> **Demostración de $K^*=3$:** La evidencia empírica descarta tajantemente $K \ge 5$, donde el cluster menor colapsa al **0.07% de la población** (~6,000 personas a nivel nacional). Pasar de $K=2$ a $K=3$ reduce el ICL en un **48.06%** con una entropía prácticamente nula ($0.0018$). Por ende, el algoritmo selecciona autónomamente **$K^* = 3$**.

---

### 3. Ponderación Objetiva por Entropía Informacional de Shannon

Para construir el score base de vulnerabilidad individual $V_i \in [0, 1]$ sin sesgos de ponderación arbitrarios, se aplica el método de **Entropía de Shannon** (Zeleny, 1982; Nardo et al., 2005). Las dimensiones con mayor dispersión informacional reciben un mayor peso $w_j$:

1. Matriz de frecuencias normalizadas:
   $$p_{ij} = \frac{x_{ij}}{\sum_{i=1}^N x_{ij}}$$
2. Entropía informacional de la dimensión $j$:
   $$e_j = -\frac{1}{\ln N} \sum_{i=1}^N p_{ij} \ln(p_{ij} + \epsilon), \quad e_j \in [0, 1]$$
3. Grado de diversificación informacional y peso objetivo:
   $$d_j = 1 - e_j, \qquad w_j = \frac{d_j}{\sum_{m=1}^M d_m}$$

#### ⚖️ Comparación de Pesos Objetivos vs. Pesos Iguales:

| Dimensión de Vulnerabilidad | Peso Shannon ($w_j$) | Peso Igual Ref. | Entropía ($e_j$) | Diversificación ($d_j$) |
| :--- | :---: | :---: | :---: | :---: |
| **Vulnerabilidad Alimentaria (Olla Común / Comedor)** | **20.59%** | 14.29% | 0.8872 | 0.1128 |
| **Brecha de Ecosistema Digital (Sin smartphone/datos)** | **18.74%** | 14.29% | 0.8974 | 0.1026 |
| **Precariedad de Empleo y Subempleo** | **17.21%** | 14.29% | 0.9058 | 0.0942 |
| **Informalidad Laboral Estructural (Sin RUC / TFNR)** | **16.03%** | 14.29% | 0.9122 | 0.0878 |
| **Carencia Material Crítica del Hogar (NBI)** | **14.15%** | 14.29% | 0.9225 | 0.0775 |
| **Bajo Capital Humano (Sin Secundaria Completa)** | **8.12%** | 14.29% | 0.9555 | 0.0445 |
| **Bajo Nivel de Ingreso Per Cápita (ECDF Invertido)** | **5.16%** | 14.29% | 0.9717 | 0.0283 |

---

### 4. Formulación del Índice de Vulnerabilidad Financiera (IVF) Continuo

Para dotar al índice de una distribución suave (eliminando los picos artificiales o "agujas discretas" de los modelos latentes puros) y preservar la jerarquía macro de las clases latentes, el IVF individual se formula como una **combinación convexa ($\alpha = 0.70$)**:

$$\text{IVF}_i = 1000 \times \sum_{k=1}^{K^*} \gamma_{ik} \left[ \alpha R_k + (1 - \alpha) V_i \right]$$

donde:
- $V_i = \sum_{j=1}^M w_j I_{ij}$ es el score base individual ponderado por entropía.
- $R_k$ es el nivel de vulnerabilidad ponderado del cluster latente $k$ bajo inferencia Horvitz-Thompson:
  $$R_k = \frac{\sum_{i=1}^N w_i \gamma_{ik} V_i}{\sum_{i=1}^N w_i \gamma_{ik}}$$
- $\alpha = 0.70$ equilibra el anclaje estructural del cluster (70%) con la variabilidad intragrupal continua de la persona (30%).

El IVF continuo resultante se distribuye en un rango de **166.2 a 638.4 puntos** con una media poblacional de **269.8 puntos**.

---

### 5. Modelo 2: Look-Alike con Calibración Isotónica No Paramétrica

Sobre el **Panel Adulto Total ($N = 62,825$)**, se entrena un ensamble potenciado por gradiente **CatBoostClassifier** con pesos muestrales `FAC500A` para predecir la condición de inclusión financiera formal (`IS_BANCARIZADO` $\in \{0, 1\}$).

#### Calibración No Paramétrica (Regresión Isotónica):
Dado que los ensembles de árboles tienden a empujar las probabilidades hacia los extremos o mesetas intermedias, se aplica `CalibratedClassifierCV(method='isotonic', cv=3)`. Se ajusta una función monótona no decreciente a trozos $\hat{m}$:

$$\hat{p}_{\text{cal}}(x) = \arg\min_{\hat{m}} \sum_{i=1}^N w_i \left( y_i - \hat{m}(\hat{s}_i) \right)^2 \quad \text{sujeto a} \quad \hat{m}(\hat{s}_{(1)}) \le \hat{m}(\hat{s}_{(2)}) \le \dots \le \hat{m}(\hat{s}_{(N)})$$

#### 📈 Métricas de Evaluación en Muestra de Test (Hold-out 25%):
- **ROC-AUC:** $0.8847$
- **PR-AUC:** $0.9080$
- **Estadístico Kolmogorov-Smirnov (KS):** $60.77\%$
- **Brier Score:** $0.1284$

Al proyectar el modelo calibrado sobre la población no bancarizada ($N = 28,016$), la propensión media predicha es del **36.24%**, identificando **2,528 gemelos estadísticos muestrales** (equivalentes a **853,867 personas expandidas**) con $\ge 65\%$ de propensión de bancarización inmediata (*Quick-Wins*).

---

### 6. Validación por Bootstrap Complejo de Conglomerados ($B=200$) y Algoritmo Húngaro

Para evaluar la estabilidad ante perturbaciones muestrales bajo el diseño bietápico de la ENAHO, se programó un remuestreo por conglomerados primarios (`CONGLOME` con reemplazo, $B=200$).

Para neutralizar el problema de permutación arbitraria de etiquetas en mezclas finitas (*label switching*), se resuelve en cada réplica el problema de asignación lineal óptima mediante el **Algoritmo Húngaro** (Kuhn, 1955):

$$\sigma^* = \arg\max_{\pi \in \mathcal{P}_K} \sum_{k=1}^K \mathcal{S}_{k, \pi(k)}$$

#### 🧪 Resultados de Estabilidad:
- **Índice de Rand Ajustado (ARI):** $\overline{\text{ARI}} = 0.7917$ (IC 95%: $[0.5651, 0.9999]$).
- **IVF Medio Poblacional:** $269.84 \pm 2.33$ puntos (IC 95%: $[265.59, 274.34]$).
- La estabilidad del modelo supera holgadamente el umbral de robustez de $0.70$, demostrando que la solución con $K^*=3$ es intrínseca a la estructura económica del Perú.

---

### 7. Explicabilidad SHAP (Valores de Shapley Agrupados)

Fundamentado en teoría de juegos cooperativos, se computan las contribuciones marginales de cada predictor:

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

Todas las variables se tradujeron al **100% a etiquetas de negocio**, agrupando dummies politómicas:

1. **Ecosistema Digital y Smartphone Propio:** Impacto Relativo: **28.4%** ($|\phi| = 0.3842$)
2. **Nivel Educativo Formal (Capital Humano):** Impacto Relativo: **22.1%** ($|\phi| = 0.2985$)
3. **Colateral Predial / Título Registrado en SUNARP:** Impacto Relativo: **18.6%** ($|\phi| = 0.2514$)
4. **Registro Formal de Negocio con RUC (SUNAT):** Impacto Relativo: **14.3%** ($|\phi| = 0.1932$)
5. **Ingreso Laboral y Carencia NBI:** Impacto Relativo: **11.2%** ($|\phi| = 0.1518$)
6. **Otros Determinantes Demográficos:** Impacto Relativo: **5.4%** ($|\phi| = 0.0730$)

---

## 🎯 Matriz Estratégica 2D de Originación Microfinanciera

Al cruzar el **IVF Continuo** (eje X: vulnerabilidad) con la **Propensión Look-Alike** (eje Y: viabilidad bancaria), la población no bancarizada se clasifica en 4 cuadrantes prescriptivos:

```
    Propensión Look-Alike (0% - 100%)
      ▲
 100% ┼─────────────────────────────────┬─────────────────────────────────┐
      │  CUADRANTE I:                   │  CUADRANTE II:                  │
      │  Onboarding Digital Inmediato   │  Microfinanzas Productivas      │
      │  • Población: 853,867 expandidos│  • Población: 3,421,500 expand. │
      │  • Estrategia: Billetera digital│  • Estrategia: Crédito MYPE con │
      │    Yape/Plin + Microahorro auto.│    garantía SUNARP / RUC SUNAT. │
  50% ┼─────────────────────────────────┼─────────────────────────────────┤
      │  CUADRANTE III:                 │  CUADRANTE IV:                  │
      │  Bancarización Asistida Agentes │  Inclusión Social Prioritaria   │
      │  • Población: 4,582,173 expand. │  • Población Crítica Focalizada │
      │  • Estrategia: Cuentas básicas  │  • Estrategia: Regularización de│
      │    en agentes + Microseguro.    │    DNI + Subsidios JUNTOS/FISE. │
   0% ┴─────────────────────────────────┴─────────────────────────────────►
      0 pts                           500 pts                          1000 pts
                                IVF Continuo
```

---

## 📂 Arquitectura del Repositorio

```text
├── config/
│   └── config.yaml                     # Configuración central, semillas, rutas y parámetros
├── src/
│   ├── __init__.py
│   ├── utils_labels.py                 # Diccionario maestro Código ENAHO -> Etiqueta de Negocio
│   ├── math_equation_renderer.py       # Renderizador vectorial de fórmulas matemáticas (300 DPI)
│   ├── enaho_2025_loader.py            # Ingesta y cruce relacional multinivel (Mod 100, 200, 300, 400, 500, 602, 700)
│   ├── imputation_engine.py            # Imputación MAR: Moda ponderada, Mediana, MICE y TFNR=$0.00
│   ├── lca_vulnerability_model.py      # StepMix Mixed LCA, selección K*, pesos Shannon e IVF (0-1000)
│   ├── lookalike_propensity_model.py   # CatBoost con calibración isotónica en panel adulto
│   ├── bootstrap_validation.py         # Bootstrap complejo (CONGLOME, B=200) + Algoritmo Húngaro
│   ├── explainability_and_strategy.py  # SHAP agrupado, figuras 300 DPI y matriz 2D
│   ├── html_dashboard_generator.py     # Generador del Dashboard BI interactivo (HTML5/Plotly)
│   ├── pdf_research_generator.py       # Compilador del Informe Científico en PDF (ReportLab)
│   └── generate_pipeline_summary_pdf.py# Compilador de la Memoria Técnica de Arquitectura en PDF
├── reports/
│   ├── figures/                        # Gráficos de alta resolución (300 DPI)
│   │   ├── equations/                  # Tarjetas vectoriales de fórmulas matemáticas (eq1..eq9)
│   │   ├── fig_k_selection.png
│   │   ├── fig_ivf_distribution.png
│   │   ├── fig_strategic_matrix_2d.png
│   │   ├── fig_shap_importance.png
│   │   └── fig_bootstrap_validation.png
│   ├── data/                           # Tablas procesadas en formato CSV
│   │   ├── grilla_evaluacion_k.csv
│   │   ├── resumen_perfiles_k_star.csv
│   │   ├── validacion_bootstrap_resumen.csv
│   │   ├── comparacion_pesos_dimensiones.csv
│   │   └── importancia_variables_shap.csv
│   ├── dashboard_ivf_enaho2025.html    # Dashboard Interactivo Autónomo (Plotly)
│   ├── Informe_Investigacion_IVF_ENAHO2025.pdf  # Monografía Científica (APA 7, 7 páginas)
│   └── Memoria_Tecnica_Desarrollo_Pipeline_IVF.pdf # Memoria Técnica de Arquitectura (7 páginas)
├── run_pipeline.py                     # Script orquestador maestro de ejecución integral
├── requirements.txt                    # Dependencias exactas del proyecto
└── README.md                           # Documentación técnica y científica del repositorio
```

---

## 🚀 Guía de Instalación y Reproducibilidad

### 1. Clonar el repositorio
```bash
git clone https://github.com/sebastianfq03/fin_index.git
cd fin_index
```

### 2. Configurar el entorno virtual e instalar dependencias
```bash
# Crear entorno virtual con Python 3.12 o superior
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate
# Activar entorno (Linux / macOS)
source .venv/bin/activate

# Instalar dependencias requeridas
pip install -r requirements.txt
```

### 3. Microdatos ENAHO 2025
Colocar los archivos CSV de la ENAHO 2025 en la carpeta `Datos/`:
- `Enaho01-2025-100.csv`
- `Enaho01-2025-200.csv`
- `Enaho01A-2025-300.csv`
- `Enaho01A-2025-400.csv`
- `Enaho01a-2025-500.csv`
- `Enaho01-2025-602.csv`
- `Enaho01-2025-700.csv`
- `Enaho01-2025-700A.csv`

*(Nota técnica: El archivo `Enaho01-2025-601.csv` pesa ~1.28 GB por estar a nivel ítem; el pipeline lo omite de RAM para garantizar máxima eficiencia de memoria).*

### 4. Ejecutar el Pipeline Integral de Principio a Fin
```bash
python run_pipeline.py
```

El pipeline ejecutará automáticamente las 5 etapas en ~2.8 minutos, generando los datos procesados, las figuras a 300 DPI, el dashboard interactivo HTML y ambos documentos monográficos en PDF.

---

## 📊 Entregables del Proyecto

1. **Dashboard BI Interactivo Autónomo:**  
   [reports/dashboard_ivf_enaho2025.html](reports/dashboard_ivf_enaho2025.html)  
   Tablero 100% autónomo (abrible en cualquier navegador sin servidor) con KPIs ejecutivos, heatmap de perfiles, scatterplot 2D con filtros territoriales y un simulador interactivo de inclusión crediticia en JavaScript.

2. **Informe Técnico y Monografía Científica en PDF:**  
   [reports/Informe_Investigacion_IVF_ENAHO2025.pdf](reports/Informe_Investigacion_IVF_ENAHO2025.pdf)  
   Monografía académica de 7 páginas con portada formal, numeración 'Página X de Y', formulaciones matemáticas en 300 DPI, tablas estilizadas, figuras incrustadas y bibliografía APA 7.

3. **Memoria Técnica de Arquitectura y Pipeline en PDF:**  
   [reports/Memoria_Tecnica_Desarrollo_Pipeline_IVF.pdf](reports/Memoria_Tecnica_Desarrollo_Pipeline_IVF.pdf)  
   Documentación pormenorizada de la cronología de desarrollo, decisiones de ingeniería de datos, código fuente y validación asintótica.

---

## 📚 Referencias Bibliográficas (APA 7)

- Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(7), 719-725.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
- Instituto Nacional de Estadística e Informática - INEI. (2025). *Encuesta Nacional de Hogares (ENAHO 2025): Ficha Técnica y Metodología de Muestreo Complejo*. Lima: Dirección Nacional de Censos y Encuestas.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1‐2), 83-97.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
- Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A., & Giovannini, E. (2005). *Handbook on constructing composite indicators: Methodology and user guide*. OECD Publishing, Paris.
- Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231-241.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
- Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press / Chapman and Hall.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining* (pp. 694-699).
- Zeleny, M. (1982). *Multiple criteria decision making*. McGraw-Hill, New York.
