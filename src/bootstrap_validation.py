"""
================================================================================
MÓDULO: src/bootstrap_validation.py
ETAPA 5: BOOTSTRAP COMPLEJO POR CONGLOMERADOS Y ALINEACIÓN HÚNGARA
================================================================================
Fundamentación Teórica y Econométrica:
--------------------------------------------------------------------------------
1. Inferencia Asintótica bajo Diseño Muestral Complejo (Cluster Bootstrap):
   La ENAHO no es una muestra aleatoria simple (M.A.S.); presenta correlación
   intra-conglomerado (efecto de diseño DEFF > 1). El remuestreo simple i.i.d.
   subestimaría drásticamente los errores estándar de los parámetros del modelo
   de clases latentes. El Bootstrap por Conglomerados (Rao & Wu, 1988):
       \\{ \\mathcal{C}_1^*, \\mathcal{C}_2^*, \\dots, \\mathcal{C}_G^* \\} \\sim \\text{Multinomial}(G; 1/G, \\dots, 1/G)
   remuestrea con reemplazo conglomerados completos (CONGLOME), preservando la
   estructura de covarianza intra-cluster de los hogares y personas.

2. Problema de Intercambio de Etiquetas (Label Switching Problem):
   Los modelos de mezclas finitas son invariantes ante permutaciones del orden
   de los componentes latentes. En cada réplica b in {1, ..., B}, el algoritmo EM
   puede rotar las etiquetas de las clases latentes. Para alinear las clases
   de cada réplica con la solución original se resuelve el problema de asignación
   lineal bipartita mediante el Algoritmo Húngaro (Kuhn, 1955):
       \\max_{\\sigma \\in \\mathcal{S}_K} \\sum_{k=1}^K n_{k, \\sigma(k)}^{(b)} \\iff \\min_{\\sigma} \\sum_{k=1}^K - n_{k, \\sigma(k)}^{(b)}
   donde N^{(b)} = [n_{k, l}^{(b)}] es la matriz de contingencia entre las asignaciones
   originales y las de la réplica b.

3. Métricas de Estabilidad e Intervalos de Confianza (95% CI):
   - Adjusted Rand Index (ARI, Hubert & Arabie, 1985): Cuantifica la concordancia
     de partición corregida por azar. Un ARI promedio >= 0.70 certifica una alta
     estabilidad y robustez de los perfiles frente a variaciones muestrales.
   - Intervalos de Confianza al 95%: Obtenidos por el método de percentiles empíricos
     [\\hat{\\theta}_{0.025}^*, \\hat{\\theta}_{0.975}^*] para el IVF promedio,
     la vulnerabilidad R_k y las masas poblacionales \\pi_k.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix

logger = logging.getLogger(__name__)


def run_complex_cluster_bootstrap(
    df_unbanked: pd.DataFrame,
    best_model: StepMix,
    k_star: int,
    cat_cols: List[str],
    cont_cols: List[str],
    config: Dict[str, Any],
    n_replicates: int = 200
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta B = 200 réplicas de Bootstrap por Conglomerados (CONGLOME).
    En cada réplica b:
    - Remuestrea conglomerados con reemplazo.
    - Re-estima el modelo StepMix con K*.
    - Alinea etiquetas con Algoritmo Húngaro.
    - Calcula R_k^(b) e IVF_medio^(b) sobre la propia muestra remuestreada b con pesos FAC500A.
    - Retorna intervalos de confianza al 95% y distribución de ARI.
    """
    logger.info(f"=== INICIANDO VALIDACIÓN BOOTSTRAP POR CONGLOMERADOS (B = {n_replicates}) ===")
    seed = config["project"]["seed"]
    cluster_col = config["bootstrap"]["cluster_col"]
    weight_col = config["data_specs"]["weight_col"]
    alpha = config["lca_model"]["alpha_ivf"]
    
    np.random.seed(seed)
    
    # 1. Preparación de datos base
    conglomes = df_unbanked[cluster_col].values
    unique_clusters = np.unique(conglomes)
    n_clusters = len(unique_clusters)
    
    # Índices por conglomerado para remuestreo vectorizado ultrarrápido
    cluster_to_indices = df_unbanked.groupby(cluster_col).indices
    
    # Matriz X y pesos
    X_cont = df_unbanked[cont_cols].values.astype(float)
    X_cont_std = (X_cont - np.mean(X_cont, axis=0)) / (np.std(X_cont, axis=0) + 1e-8)
    X_cat = df_unbanked[cat_cols].values.astype(int)
    X_all = np.hstack([X_cat, X_cont_std])
    weights = df_unbanked[weight_col].values.astype(float)
    V_i = df_unbanked["score_vulnerabilidad_base"].values.astype(float)
    
    # Asignación original de clases
    orig_probs = best_model.predict_proba(X_all)
    orig_labels = np.argmax(orig_probs, axis=1)
    
    measurement_structure = {
        "cat": {"model": "bernoulli", "n_columns": len(cat_cols)},
        "cont": {"model": "gaussian_diag", "n_columns": len(cont_cols)}
    }
    
    # Contenedores para estadísticas bootstrap
    b_mean_ivf: List[float] = []
    b_ari_scores: List[float] = []
    b_rk: List[np.ndarray] = []
    b_masses: List[np.ndarray] = []
    
    for b in range(n_replicates):
        # 1. Remuestreo de conglomerados con reemplazo
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        b_idx = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        
        # Muestra b
        X_b = X_all[b_idx]
        w_b = weights[b_idx]
        V_b = V_i[b_idx]
        
        # 2. Re-estimación del modelo StepMix con K*
        b_model = StepMix(
            n_components=k_star,
            measurement=measurement_structure,
            random_state=seed + b + 1,
            verbose=0,
            progress_bar=0,
            max_iter=100,
            n_init=1
        )
        try:
            b_model.fit(X_b, sample_weight=w_b)
        except Exception:
            continue
            
        # Predicción sobre muestra bootstrap b
        gamma_b = b_model.predict_proba(X_b)
        labels_b = np.argmax(gamma_b, axis=1)
        
        # Predicción sobre muestra original para evaluación de ARI y contingencia
        gamma_b_on_orig = b_model.predict_proba(X_all)
        labels_b_on_orig = np.argmax(gamma_b_on_orig, axis=1)
        
        # 3. Alineación Húngara de Etiquetas
        # Matriz de costos negativa basada en la contingencia
        cost_matrix = np.zeros((k_star, k_star))
        for i in range(k_star):
            for j in range(k_star):
                cost_matrix[i, j] = -np.sum((orig_labels == i) & (labels_b_on_orig == j))
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Mapeo de etiqueta bootstrap a etiqueta original
        mapping = {boot_k: orig_k for orig_k, boot_k in zip(row_ind, col_ind)}
        
        # Alinear probabilidades y etiquetas en la muestra b
        gamma_b_aligned = np.zeros_like(gamma_b)
        labels_b_aligned = np.zeros_like(labels_b)
        for orig_k, boot_k in zip(row_ind, col_ind):
            gamma_b_aligned[:, orig_k] = gamma_b[:, boot_k]
            labels_b_aligned[labels_b == boot_k] = orig_k
            
        # Alinear etiquetas en la muestra original para ARI
        labels_b_orig_aligned = np.zeros_like(labels_b_on_orig)
        for orig_k, boot_k in zip(row_ind, col_ind):
            labels_b_orig_aligned[labels_b_on_orig == boot_k] = orig_k
            
        ari_val = adjusted_rand_score(orig_labels, labels_b_orig_aligned)
        b_ari_scores.append(ari_val)
        
        # 4. Cálculo de R_k^(b) e IVF_medio^(b) sobre la propia muestra remuestreada b
        R_k_b = np.zeros(k_star)
        for k in range(k_star):
            num = np.sum(w_b * gamma_b_aligned[:, k] * V_b)
            den = np.sum(w_b * gamma_b_aligned[:, k]) + 1e-12
            R_k_b[k] = num / den
        b_rk.append(R_k_b)
        
        # Masas poblacionales en b
        mass_b = np.sum(gamma_b_aligned * w_b[:, None], axis=0) / np.sum(w_b)
        b_masses.append(mass_b)
        
        # IVF continuo en muestra b
        comp_clase_b = np.dot(gamma_b_aligned, R_k_b)
        ivf_b = 1000.0 * (alpha * comp_clase_b + (1.0 - alpha) * V_b)
        mean_ivf_b = np.average(ivf_b, weights=w_b)
        b_mean_ivf.append(mean_ivf_b)
        
    logger.info(f"Bootstrap finalizado con éxito ({len(b_mean_ivf)} réplicas válidas).")
    
    # 5. Cálculo de Estadísticos e Intervalos de Confianza al 95%
    ci_level = config["bootstrap"]["confidence_level"]
    lower_pct = ((1.0 - ci_level) / 2.0) * 100
    upper_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100
    
    mean_ivf_est = float(np.mean(b_mean_ivf))
    ci_ivf_low = float(np.percentile(b_mean_ivf, lower_pct))
    ci_ivf_high = float(np.percentile(b_mean_ivf, upper_pct))
    
    mean_ari_est = float(np.mean(b_ari_scores))
    ci_ari_low = float(np.percentile(b_ari_scores, lower_pct))
    ci_ari_high = float(np.percentile(b_ari_scores, upper_pct))
    
    # Tabla resumen de validación
    res_rows = [
        {
            "Parametro": "IVF Medio Poblacional",
            "Media_Bootstrap": round(mean_ivf_est, 2),
            "Error_Estandar": round(float(np.std(b_mean_ivf)), 2),
            "IC_95_Inferior": round(ci_ivf_low, 2),
            "IC_95_Superior": round(ci_ivf_high, 2)
        },
        {
            "Parametro": "Índice de Rand Ajustado (ARI)",
            "Media_Bootstrap": round(mean_ari_est, 4),
            "Error_Estandar": round(float(np.std(b_ari_scores)), 4),
            "IC_95_Inferior": round(ci_ari_low, 4),
            "IC_95_Superior": round(ci_ari_high, 4)
        }
    ]
    
    # IC para R_k y masas de cada perfil
    b_rk_mat = np.array(b_rk)
    b_mass_mat = np.array(b_masses) * 100.0
    
    for k in range(k_star):
        res_rows.append({
            "Parametro": f"Vulnerabilidad Base Perfil R_{k}",
            "Media_Bootstrap": round(float(np.mean(b_rk_mat[:, k])), 4),
            "Error_Estandar": round(float(np.std(b_rk_mat[:, k])), 4),
            "IC_95_Inferior": round(float(np.percentile(b_rk_mat[:, k], lower_pct)), 4),
            "IC_95_Superior": round(float(np.percentile(b_rk_mat[:, k], upper_pct)), 4)
        })
        res_rows.append({
            "Parametro": f"Masa Poblacional Perfil {k} (%)",
            "Media_Bootstrap": round(float(np.mean(b_mass_mat[:, k])), 2),
            "Error_Estandar": round(float(np.std(b_mass_mat[:, k])), 2),
            "IC_95_Inferior": round(float(np.percentile(b_mass_mat[:, k], lower_pct)), 2),
            "IC_95_Superior": round(float(np.percentile(b_mass_mat[:, k], upper_pct)), 2)
        })
        
    bootstrap_summary_df = pd.DataFrame(res_rows)
    logger.info("Resumen de Validación Bootstrap por Conglomerados:\n" + bootstrap_summary_df.to_string(index=False))
    
    distribution_data = {
        "b_mean_ivf": b_mean_ivf,
        "b_ari_scores": b_ari_scores,
        "b_rk": b_rk_mat,
        "b_masses": b_mass_mat
    }
    
    return bootstrap_summary_df, distribution_data
