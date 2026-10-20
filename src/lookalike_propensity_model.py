"""
================================================================================
MÓDULO: src/lookalike_propensity_model.py
ETAPA 4: LOOK-ALIKE MODELING DE PROPENSIÓN A LA BANCARIZACIÓN
================================================================================
Fundamentación Econométrica y Teórica:
--------------------------------------------------------------------------------
1. Metodología Look-Alike (Gemelos Estadísticos en Microfinanzas):
   En economía aplicada y analítica bancaria, el modelo Look-Alike estima la
   probabilidad condicional de adopción de servicios financieros formales:
       P(IS_BANCARIZADO_i = 1 | X_i) = \\sigma(f(X_i))
   donde f(X_i) es un ensamble de árboles con gradiente potenciado (CatBoost)
   entrenado sobre el panel completo de adultos en edad activa (18 a 65 años) de la
   ENAHO 2025 bajo ponderación de diseño Horvitz-Thompson (w_i = FAC500A_i).

2. Calibración No Paramétrica de Probabilidades (Regresión Isotónica):
   Los modelos basados en árboles tienden a distorsionar las probabilidades
   en los extremos debido a la penalización de regularización y la compresión
   sigmoidal. La calibración isotónica (Zadrozny & Elkan, 2002):
       \\min_{m} \\sum_{i=1}^N w_i \\left( y_i - m(\\hat{p}_i) \\right)^2 \\quad \\text{s.t. } m(p_1) \\le m(p_2) \\text{ si } p_1 \\le p_2
   ajusta una función monótona no decreciente a trozos mediante validación cruzada (cv=3),
   transformando el score crudo en una verdadera probabilidad a posteriori libre de sesgo.

3. Inferencia de Gemelos Estadísticos en la Población No Bancarizada:
   Al aplicar el modelo calibrado sobre la población no bancarizada objetivo:
       PROPENSITY_SCORE_i = m(\\hat{p}_i) \\in [0, 1]
   se identifica a aquellos individuos no bancarizados cuyos vectores de capital humano,
   ingresos, conectividad digital y garantías patrimoniales replican fielmente
   el perfil de los clientes formales del sistema financiero ('Quick-Wins').

4. Métricas de Evaluación para Desbalance Moderado:
   - ROC-AUC: Capacidad de ordenamiento discriminante global.
   - PR-AUC: Área bajo la curva Precision-Recall, sensible al desbalance de clases.
   - Estadístico de Kolmogorov-Smirnov (KS):
         KS = \\max_t | F_1(t) - F_0(t) |
     que mide la máxima separación acumulada entre bancarizados y no bancarizados.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import ks_2samp
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def prepare_lookalike_matrices(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepara la matriz de características X con codificación dummy para variables
    politómicas (dominio_macro), asegurando compatibilidad con scikit-learn y SHAP.
    """
    base_features = [
        "edad", "educacion_anios", "log_ingreso",
        "tiene_ruc", "tiene_colateral_propiedad", "tiene_plan_datos",
        "ventas_digitales", "nbi_score", "es_urbano", "acceso_digital",
        "es_mujer", "es_jefe_hogar", "informalidad", "empleo_vulnerable",
        "ahorro_juntas", "ahorro_casa", "dominio_macro"
    ]
    
    cols_to_use = [c for c in base_features if c in df.columns]
    X_raw = df[cols_to_use].copy()
    
    # One-hot encoding de categorías
    if "dominio_macro" in X_raw.columns:
        X_encoded = pd.get_dummies(X_raw, columns=["dominio_macro"], drop_first=False, dtype=float)
    else:
        X_encoded = X_raw.astype(float)
        
    final_cols = X_encoded.columns.tolist()
    
    # Si se especificó una lista de columnas de entrenamiento previa (e.g. para scoring), alinear
    if feature_cols is not None:
        for c in feature_cols:
            if c not in X_encoded.columns:
                X_encoded[c] = 0.0
        X_encoded = X_encoded[feature_cols]
        final_cols = feature_cols
        
    return X_encoded, final_cols


def train_and_calibrate_lookalike_model(
    df_total: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[Any, Any, Dict[str, float], List[str]]:
    """
    Entrena y calibra el modelo supervisado CatBoost Look-Alike sobre el panel
    total adulto de ENAHO 2025 con ponderación FAC500A.
    """
    logger.info("=== ENTRENANDO MODELO LOOK-ALIKE DE PROPENSIÓN A LA BANCARIZACIÓN ===")
    seed = config["project"]["seed"]
    weight_col = config["data_specs"]["weight_col"]
    target_col = config["lookalike_model"]["target_col"]
    test_size = config["lookalike_model"]["test_size"]
    
    X, features = prepare_lookalike_matrices(df_total)
    y = df_total[target_col].values.astype(int)
    w = df_total[weight_col].values.astype(float)
    logger.info(f"Matriz de entrenamiento preparada: {X.shape[0]:,} observaciones × {X.shape[1]} variables.")
    
    # Split estratificado con pesos
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    
    # Estimador base CatBoost
    base_cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.07,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=0
    )
    
    # Ajuste del modelo base para cálculo de SHAP
    base_cb.fit(X_train, y_train, sample_weight=w_train)
    
    # Calibrador Isotónico mediante 3-fold cross validation
    logger.info("Entrenando ensamble calibrado mediante Regresión Isotónica (cv=3)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_cb,
        method="isotonic",
        cv=3
    )
    calibrated_model.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluación sobre el conjunto de prueba retenido
    calib_test_preds = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Métricas ponderadas
    roc_auc = float(roc_auc_score(y_test, calib_test_preds, sample_weight=w_test))
    pr_auc = float(average_precision_score(y_test, calib_test_preds, sample_weight=w_test))
    brier = float(brier_score_loss(y_test, calib_test_preds, sample_weight=w_test))
    ks_stat = float(ks_2samp(calib_test_preds[y_test == 1], calib_test_preds[y_test == 0]).statistic)
    
    metrics = {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Brier_Score": round(brier, 4),
        "KS_Statistic": round(ks_stat * 100, 2),
        "N_Train": int(len(X_train)),
        "N_Test": int(len(X_test))
    }
    
    logger.info(
        f"Métricas del Modelo Look-Alike en Test: ROC-AUC = {metrics['ROC_AUC']:.4f} | "
        f"PR-AUC = {metrics['PR_AUC']:.4f} | KS = {metrics['KS_Statistic']:.2f}% | "
        f"Brier = {metrics['Brier_Score']:.4f}"
    )
    
    return calibrated_model, base_cb, metrics, features


def score_unbanked_population(
    calibrated_model: Any,
    base_model: Any,
    df_unbanked: pd.DataFrame,
    features: List[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calcula el Score de Propensión a la Bancarización (0% a 100%) sobre la
    población no bancarizada objetivo de ENAHO 2025.
    """
    logger.info("Asignando Scores de Propensión Look-Alike a la Población No Bancarizada...")
    df_unb = df_unbanked.copy()
    X_unb, _ = prepare_lookalike_matrices(df_unb, feature_cols=features)
    
    propensity = calibrated_model.predict_proba(X_unb)[:, 1]
    propensity = np.clip(propensity, 0.0, 1.0)
    
    df_unb["propensity_score"] = np.round(propensity, 4)
    df_unb["propensity_pct"] = np.round(propensity * 100, 2)
    
    logger.info(
        f"Scoring finalizado. Propensión Media: {np.mean(propensity)*100:.2f}% "
        f"(Quick-Wins >= 65%: {(propensity >= 0.65).mean()*100:.2f}% de la población objetivo)."
    )
    
    return df_unb, propensity
