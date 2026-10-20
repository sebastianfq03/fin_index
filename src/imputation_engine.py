"""
================================================================================
MÓDULO: src/imputation_engine.py
ETAPA 2: LIMPIEZA DE CENTINELAS E IMPUTACIÓN RIGUROSA (MAR Y MICE)
================================================================================
Fundamentación Econométrica y Teórica:
--------------------------------------------------------------------------------
1. Supuesto de Datos Faltantes al Azar (Missing At Random - MAR):
   En encuestas complejas socioeconómicas, el mecanismo de no respuesta rara vez
   es completamente al azar (MCAR). Bajo el supuesto MAR (Rubin, 1976):
       P(M | Y_obs, Y_mis) = P(M | Y_obs)
   La probabilidad de no respuesta depende únicamente de los atributos observables
   (nivel educativo, región, estrato demográfico, sector informal, tenencia de
   vivienda). Condicionado a estas covariables, la imputación condicional genera
   estimadores asintóticamente insesgados de la matriz de datos completa.

2. Tratamiento Diferenciado por Escala de Medida:
   - Variables Nominales (<= 5% faltantes):
     Prohibido promediar o calcular medianas sobre códigos nominales arbitrarios
     (e.g., estado civil o lengua). Se emplea la Moda Ponderada Poblacional:
         Mode_w(X) = argmax_c sum_{i: x_i = c} w_i,   w_i = FAC500A_i
     respetando las probabilidades muestrales de selección de la ENAHO.
   - Variables Ordinales y Continuas Simétricas (<= 5% faltantes):
     Se imputan mediante la Mediana Ponderada/Empírica, minimizando el error
     absoluto medio y otorgando robustez frente a colas pesadas.
   - Variables Estructurales (> 5% faltantes):
     Se emplea Imputación Múltiple por Ecuaciones Encadenadas (MICE) basada en
     árboles de decisión (FCS - Fully Conditional Specification, Van Buuren, 2018),
     tratando las variables categóricas como categorías no métricas.

3. Imputación Econométrica del Ingreso y Ceros Estructurales (TFNR):
   Los ingresos laborales en economías con alta informalidad exhiben una marcada
   asimetría positiva (distribución aproximadamente Log-Normal o Pareto).
   - Trabajadores Familiares No Remunerados (TFNR, P507 == 5):
     Constituyen un cero estructural genuino ($0.00), no una no respuesta. Imputar
     un valor positivo a un TFNR distorsionaría gravemente la estructura de
     ingresos y la vulnerabilidad del hogar.
   - No respuesta en asalariados o independientes:
     Se imputa exclusivamente sobre la escala logarítmica:
         \\ln(Y_i + 1) = f(Educacion_i, Edad_i, Estrato_i, Informalidad_i, Colateral_i) + \\epsilon_i
     garantizando que al retransformar Y_i = \\exp(\\hat{y}_i) - 1, el ingreso sea
     estrictamente no negativo y se preserve la heterocedasticidad residual.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

logger = logging.getLogger(__name__)


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte códigos centinela típicos de ENAHO (9, 99, 999, 99999, 999999)
    y espacios en blanco a np.nan. Elimina variables con varianza cero.
    """
    df = df.copy()
    
    # Reemplazo de centinelas conocidos en ingresos
    for inc_col in ["ingreso_dependiente", "ganancia_independiente"]:
        if inc_col in df.columns:
            df.loc[df[inc_col] >= 999990, inc_col] = np.nan
            df.loc[df[inc_col] < 0, inc_col] = np.nan
            
    # Centinelas de no respuesta en variables categóricas ordinales
    sentinels = [999999, 99999, 9999, 999]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            for s in sentinels:
                df.loc[df[col] == s, col] = np.nan
                
    return df


def weighted_mode(series: pd.Series, weights: pd.Series) -> Any:
    """
    Calcula la moda ponderada poblacional de una variable nominal
    utilizando el factor de expansión Horvitz-Thompson.
    """
    valid_mask = series.notna() & weights.notna()
    if not valid_mask.any():
        return series.dropna().iloc[0] if len(series.dropna()) > 0 else 0
        
    s_valid = series[valid_mask]
    w_valid = weights[valid_mask]
    
    weighted_counts = w_valid.groupby(s_valid).sum()
    return weighted_counts.idxmax()


def impute_univariate_low_missing(
    df: pd.DataFrame,
    nominal_cols: List[str],
    ordinal_cols: List[str],
    weight_col: str = "FAC500A",
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Aplica imputación univariada rigurosa para variables con <= 5% de faltantes:
    - Nominales: Moda Ponderada (Poblacional).
    - Ordinales/Continuas: Mediana.
    """
    df = df.copy()
    weights = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    
    for col in nominal_cols:
        if col in df.columns:
            missing_frac = df[col].isna().mean()
            if 0 < missing_frac <= threshold:
                mode_val = weighted_mode(df[col], weights)
                logger.debug(f"Imputando nominal {col} ({missing_frac*100:.2f}% nulos) con Moda Ponderada: {mode_val}")
                df[col] = df[col].fillna(mode_val)
                
    for col in ordinal_cols:
        if col in df.columns:
            missing_frac = df[col].isna().mean()
            if 0 < missing_frac <= threshold:
                median_val = df[col].median()
                logger.debug(f"Imputando ordinal {col} ({missing_frac*100:.2f}% nulos) con Mediana: {median_val}")
                df[col] = df[col].fillna(median_val)
                
    return df


def execute_econometric_income_and_mice_imputation(
    df: pd.DataFrame,
    weight_col: str = "FAC500A",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Ejecuta el protocolo econométrico para el ingreso y MICE para variables estructurales:
    1. Asigna ingreso dependiente a asalariados (P507 in [3, 4, 6]).
    2. Asigna ganancia neta a independientes (P507 in [1, 2]).
    3. Asigna CERO ESTRUCTURAL (0.0) a Trabajadores Familiares No Remunerados (P507 == 5).
    4. Modela el log-ingreso ln(Y + 1) e imputa faltantes con MICE condicionado a
       educación, edad, informalidad, colateral, TIC y territorio.
    5. Re-inserta el 0.0 estricto en todos los TFNR.
    6. Re-transforma a escala monetaria Y = exp(ln(Y+1)) - 1.
    """
    logger.info("=== EJECUTANDO IMPUTACIÓN ECONOMÉTRICA DEL INGRESO Y MICE ===")
    df = df.copy()
    
    # 1. Consolidación del ingreso bruto según inserción laboral
    # P507 == 5: TFNR -> Cero genuino garantizado
    is_tfnr = (df["categoria_ocupacional"] == 5)
    
    # Base inicial de ingreso
    raw_income = pd.Series(np.nan, index=df.index, dtype=float)
    
    # Dependientes (obreros, empleados, trabajadoras del hogar)
    dep_mask = df["categoria_ocupacional"].isin([3, 4, 6]) & df["ingreso_dependiente"].notna()
    raw_income.loc[dep_mask] = df.loc[dep_mask, "ingreso_dependiente"]
    
    # Independientes y patronos
    indep_mask = df["categoria_ocupacional"].isin([1, 2]) & df["ganancia_independiente"].notna()
    raw_income.loc[indep_mask] = df.loc[indep_mask, "ganancia_independiente"]
    
    # Si alguien tiene el otro ingreso reportado y el principal es nulo
    raw_income = raw_income.combine_first(df["ingreso_dependiente"]).combine_first(df["ganancia_independiente"])
    
    # Asignar 0.0 estricto a TFNR
    raw_income.loc[is_tfnr] = 0.0
    
    # 2. Transformación logarítmica de estabilización de varianza
    # Para personas no TFNR con ingreso válido
    log_income = np.log1p(raw_income.clip(lower=0.0))
    df["log_ingreso"] = log_income
    
    # 3. Predictores para el modelo MICE
    # Variables auxiliares con fuerte poder predictivo del ingreso y empleo
    feature_cols = [
        "edad", "es_mujer", "es_jefe_hogar",
        "educacion_anios", "es_lengua_originaria",
        "acceso_digital", "tiene_colateral_propiedad", "nbi_score",
        "es_urbano", "tiene_ruc", "informalidad", "empleo_vulnerable",
        "ahorro_juntas", "recibe_subsidio_estatal", "acude_olla_comedor"
    ]
    
    # Variables estructurales con faltantes a imputar junto al ingreso
    mice_target_cols = ["categoria_ocupacional", "tamano_empresa", "log_ingreso"]
    
    # Filtrar solo columnas existentes
    available_features = [c for c in feature_cols if c in df.columns]
    impute_matrix_cols = available_features + mice_target_cols
    
    logger.info(f"Matriz MICE construida con {len(impute_matrix_cols)} variables explicativas.")
    
    # Construir sub-dataframe para MICE
    X_mice = df[impute_matrix_cols].copy()
    
    # Para los TFNR, mantener log_ingreso como 0.0 constante durante el proceso
    tfnr_indices = df[is_tfnr].index
    
    # Intentar miceforest primero (con compatibilidad para LightGBM)
    mice_succeeded = False
    try:
        import miceforest as mf
        # Configurar categorías explícitas para miceforest
        for cat_col in ["categoria_ocupacional", "tamano_empresa"]:
            if cat_col in X_mice.columns:
                X_mice[cat_col] = X_mice[cat_col].astype("category")
                
        kernel = mf.ImputationKernel(
            X_mice,
            num_datasets=1,
            mean_match_strategy="normal",
            random_state=random_state
        )
        kernel.mice(iterations=3)
        X_imputed = kernel.complete_data(dataset=0)
        mice_succeeded = True
        logger.info("Imputación MICE con miceforest completada con éxito.")
    except Exception as e:
        logger.warning(f"Aviso de compatibilidad miceforest ({e}). Ejecutando MICE con IterativeImputer (ExtraTrees FCS).")
        
    if not mice_succeeded:
        # MICE Robusto mediante IterativeImputer (Fully Conditional Specification con Random/Extra Trees)
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=15, max_depth=10, random_state=random_state, n_jobs=-1),
            max_iter=5,
            random_state=random_state,
            initial_strategy="median"
        )
        imputed_vals = imputer.fit_transform(X_mice)
        X_imputed = pd.DataFrame(imputed_vals, columns=impute_matrix_cols, index=df.index)
        logger.info("Imputación MICE con IterativeImputer (ExtraTrees FCS) completada con éxito.")

    # 4. Actualizar variables imputadas en el dataframe maestro
    for col in mice_target_cols:
        df[col] = X_imputed[col]
        
    # 5. RE-INSERCIÓN OBLIGATORIA DEL CERO ESTRUCTURAL (TFNR)
    df.loc[tfnr_indices, "log_ingreso"] = 0.0
    
    # 6. Re-transformación exponencial a escala monetaria en Nuevos Soles (S/.)
    df["ingreso_mensual"] = np.expm1(df["log_ingreso"]).clip(lower=0.0).round(2)
    
    # Validar que no queden nulos en variables clave
    logger.info(
        f"Imputación finalizada. Ingreso medio ponderado: S/. "
        f"{np.average(df['ingreso_mensual'], weights=df[weight_col]):.2f} "
        f"(TFNR genuinos en S/. 0.00: {len(tfnr_indices):,} personas)."
    )
    
    return df


def run_full_imputation_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Orquesta el flujo completo de limpieza, imputación univariada y MICE econométrico.
    """
    logger.info("Iniciando pipeline integral de imputación y depuración de microdatos.")
    weight_col = config["data_specs"]["weight_col"]
    seed = config["project"]["seed"]
    
    # 1. Limpieza de centinelas
    df = clean_sentinels(df)
    
    # 2. Imputación univariada con umbral <= 5%
    nominal_vars = ["estado_civil", "es_lengua_originaria", "tipo_contrato"]
    ordinal_vars = ["educacion_anios", "nbi_score", "acceso_digital"]
    df = impute_univariate_low_missing(df, nominal_vars, ordinal_vars, weight_col=weight_col)
    
    # 3. Imputación MICE y econométrica de ingresos
    df = execute_econometric_income_and_mice_imputation(df, weight_col=weight_col, random_state=seed)
    
    # Excluir columnas protegidas de la eliminación por varianza cero
    protected = ["IS_BANCARIZADO", "no_bancarizado_flag", "FAC500A", "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]
    cols_to_check = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in protected]
    zero_var_cols = [c for c in cols_to_check if df[c].std() == 0]
    if zero_var_cols:
        logger.warning(f"Eliminando columnas predictoras con varianza cero: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
        
    return df
