"""
================================================================================
SCRIPT MAESTRO: run_pipeline.py
PROYECTO: ÍNDICE DE VULNERABILIDAD FINANCIERA (IVF) & MODELADO LOOK-ALIKE (ENAHO 2025)
================================================================================
Ejecuta de principio a fin el pipeline analítico y econométrico:
1. Etapa 1: Ingeniería de datos e integración relacional multi-nivel ENAHO 2025.
2. Etapa 2: Limpieza de centinelas e imputación rigurosa MAR + MICE (TFNR=0.00).
3. Etapa 3: Modelo 1 — Clases Latentes Mixtas (StepMix), selección autónoma de K*,
            pesos de Shannon y cálculo del IVF Continuo (0-1000).
4. Etapa 4: Modelo 2 — CatBoost Look-Alike con calibración isotónica en panel adulto
            y scoring de propensión en no bancarizados.
5. Etapa 5: Bootstrap complejo por conglomerados (CONGLOME, B=200) + Algoritmo Húngaro,
            explicabilidad SHAP agrupada y Matriz Estratégica 2D.
6. Entregables: Exportación de datos limpios, Figuras 300 DPI, Dashboard BI (HTML)
                e Informe de Investigación Científica (PDF).
"""

import os
import sys
import time
import logging
import yaml
import numpy as np
import pandas as pd

from src.enaho_2025_loader import integrate_enaho_2025_datasets
from src.imputation_engine import run_full_imputation_pipeline
from src.lca_vulnerability_model import (
    prepare_lca_feature_matrices,
    evaluate_lca_k_grid,
    compute_shannon_weights_and_base_score,
    build_continuous_ivf
)
from src.lookalike_propensity_model import (
    train_and_calibrate_lookalike_model,
    score_unbanked_population
)
from src.bootstrap_validation import run_complex_cluster_bootstrap
from src.explainability_and_strategy import (
    assign_strategic_quadrants,
    compute_shap_explanations,
    generate_research_figures
)
from src.html_dashboard_generator import generate_interactive_bi_dashboard
from src.pdf_research_generator import build_research_pdf_report
from src.generate_pipeline_summary_pdf import build_pipeline_summary_pdf

# Configurar logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_execution.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("IVF_Pipeline")


def main():
    t_start = time.time()
    logger.info("================================================================================")
    logger.info("   INICIANDO PIPELINE MAESTRO: IVF Y MODELADO LOOK-ALIKE (ENAHO 2025)           ")
    logger.info("================================================================================")
    
    # 0. Carga de Configuración
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config["paths"]["reports_dir"], exist_ok=True)
    os.makedirs(config["paths"]["figures_dir"], exist_ok=True)
    os.makedirs(config["paths"]["processed_dir"], exist_ok=True)
    
    # --------------------------------------------------------------------------
    # ETAPA 1: INGENIERÍA DE DATOS E INTEGRACIÓN ENAHO 2025
    # --------------------------------------------------------------------------
    logger.info(">>> [ETAPA 1/5] Carga e Integración Relacional Multi-Nivel de ENAHO 2025...")
    df_total, df_unbanked = integrate_enaho_2025_datasets(config)
    logger.info(f"Panel Adulto Total (18-65 años): {len(df_total):,} registros.")
    logger.info(f"Población Objetivo No Bancarizada: {len(df_unbanked):,} registros.")
    
    # --------------------------------------------------------------------------
    # ETAPA 2: LIMPIEZA DE CENTINELAS E IMPUTACIÓN ECONOMÉTRICA (MAR + MICE)
    # --------------------------------------------------------------------------
    logger.info(">>> [ETAPA 2/5] Limpieza de Centinelas e Imputación Econométrica MAR / MICE...")
    df_unbanked_clean = run_full_imputation_pipeline(df_unbanked, config)
    df_total_clean = run_full_imputation_pipeline(df_total, config)
    
    # --------------------------------------------------------------------------
    # ETAPA 3: MODELO 1 — MEZCLAS MIXTAS (STEPMIX), K* ÓPTIMO E ÍNDICE IVF
    # --------------------------------------------------------------------------
    logger.info(">>> [ETAPA 3/5] Estimación de Clases Latentes Mixtas y Búsqueda de K* Óptimo...")
    X_all, cat_cols, cont_cols, weights = prepare_lca_feature_matrices(df_unbanked_clean)
    best_model, k_star, metrics_grid_df, all_models = evaluate_lca_k_grid(
        X_all, cat_cols, cont_cols, weights, config
    )
    
    logger.info("Calculando Ponderación de Shannon y Score Base Vi...")
    V_i, shannon_w, equal_w, weights_comp_df = compute_shannon_weights_and_base_score(df_unbanked_clean)
    
    logger.info("Construyendo Índice de Vulnerabilidad Financiera (IVF Continuo 0-1000)...")
    df_unbanked_ivf, cluster_summary_df, profile_names = build_continuous_ivf(
        df_unbanked_clean, best_model, X_all, V_i, config
    )
    
    # --------------------------------------------------------------------------
    # ETAPA 4: MODELO 2 — LOOK-ALIKE DE PROPENSIÓN A LA BANCARIZACIÓN
    # --------------------------------------------------------------------------
    logger.info(">>> [ETAPA 4/5] Entrenamiento del Modelo Look-Alike (CatBoost Calibrado)...")
    calibrated_model, base_cb, lookalike_metrics, lookalike_features = train_and_calibrate_lookalike_model(
        df_total_clean, config
    )
    
    logger.info("Infiriendo Propensión a Bancarización sobre No Bancarizados...")
    df_unbanked_scored, propensity_scores = score_unbanked_population(
        calibrated_model, base_cb, df_unbanked_ivf, lookalike_features
    )
    
    # --------------------------------------------------------------------------
    # ETAPA 5: BOOTSTRAP POR CONGLOMERADOS, SHAP Y MATRIZ ESTRATÉGICA 2D
    # --------------------------------------------------------------------------
    logger.info(">>> [ETAPA 5/5] Bootstrap Complejo por Conglomerados (B=200) + SHAP + Matriz 2D...")
    n_boot = config["bootstrap"]["n_replicates"]
    bootstrap_summary_df, bootstrap_data = run_complex_cluster_bootstrap(
        df_unbanked_scored, best_model, k_star, cat_cols, cont_cols, config, n_replicates=n_boot
    )
    
    logger.info("Asignando Cuadrantes Estratégicos de Inclusión 2D...")
    df_final = assign_strategic_quadrants(df_unbanked_scored, config)
    
    logger.info("Calculando Valores de Shapley (SHAP)...")
    shap_vals, shap_importance_df = compute_shap_explanations(
        base_cb, df_total_clean, lookalike_features
    )
    
    logger.info("Generando Figuras de Investigación de Alta Resolución (300 DPI)...")
    figure_paths = generate_research_figures(
        df_final, metrics_grid_df, shap_importance_df, bootstrap_data,
        cluster_summary_df, config["paths"]["figures_dir"]
    )
    
    # --------------------------------------------------------------------------
    # EXPORTACIÓN DE RESULTADOS Y BASES PROCESADAS
    # --------------------------------------------------------------------------
    processed_dir = config["paths"]["processed_dir"]
    logger.info(f"Exportando resultados tabulares a {processed_dir}...")
    
    df_final.to_csv(os.path.join(processed_dir, "enaho_2025_no_bancarizados_ivf_scored.csv"), index=False, sep=";", encoding="latin1")
    cluster_summary_df.to_csv(os.path.join(processed_dir, "resumen_perfiles_k_star.csv"), sep=";", encoding="latin1")
    metrics_grid_df.to_csv(os.path.join(processed_dir, "grilla_evaluacion_k.csv"), index=False, sep=";", encoding="latin1")
    bootstrap_summary_df.to_csv(os.path.join(processed_dir, "validacion_bootstrap_resumen.csv"), index=False, sep=";", encoding="latin1")
    shap_importance_df.to_csv(os.path.join(processed_dir, "importancia_variables_shap.csv"), index=False, sep=";", encoding="latin1")
    weights_comp_df.to_csv(os.path.join(processed_dir, "comparacion_pesos_dimensiones.csv"), index=False, sep=";", encoding="latin1")
    
    # --------------------------------------------------------------------------
    # COMPILACIÓN DE ENTREGABLES (HTML + PDF)
    # --------------------------------------------------------------------------
    logger.info(">>> Compilando Dashboard Interactivo Autónomo (Plotly HTML)...")
    html_path = generate_interactive_bi_dashboard(
        df_final, df_total_clean, metrics_grid_df, cluster_summary_df,
        shap_importance_df, bootstrap_summary_df, profile_names, k_star,
        config["paths"]["dashboard_html"]
    )
    
    logger.info(">>> Compilando Informe Técnico y Científico de Investigación (ReportLab PDF)...")
    pdf_path = build_research_pdf_report(
        df_final, df_total_clean, metrics_grid_df, cluster_summary_df,
        shap_importance_df, bootstrap_summary_df, weights_comp_df,
        lookalike_metrics, figure_paths, k_star, config["paths"]["report_pdf"]
    )

    logger.info(">>> Compilando Memoria Técnica del Pipeline (ReportLab PDF)...")
    memoria_pdf_path = os.path.join(config["paths"]["reports_dir"], "Memoria_Tecnica_Desarrollo_Pipeline_IVF.pdf")
    build_pipeline_summary_pdf(memoria_pdf_path)
    
    t_elapsed = time.time() - t_start
    logger.info("================================================================================")
    logger.info(f"   PIPELINE COMPLETADO CON ÉXITO EN {t_elapsed/60:.2f} MINUTOS ({t_elapsed:.1f} SEGUNDOS)    ")
    logger.info("================================================================================")
    logger.info(f"1. K* Óptimo Seleccionado Algorítmicamente: K* = {k_star}")
    logger.info(f"2. Población No Bancarizada Expandida: {int(df_final['FAC500A'].sum()):,} personas")
    logger.info(f"3. IVF Medio Ponderado: {np.average(df_final['indice_ivf'], weights=df_final['FAC500A']):.2f} puntos")
    logger.info(f"4. Desempeño Look-Alike en Test: ROC-AUC = {lookalike_metrics['ROC_AUC']:.4f} | KS = {lookalike_metrics['KS_Statistic']:.2f}%")
    logger.info(f"5. Dashboard HTML Autónomo: {html_path} ({os.path.getsize(html_path):,} bytes)")
    logger.info(f"6. Informe Científico PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
    logger.info(f"7. Memoria Técnica PDF: {memoria_pdf_path} ({os.path.getsize(memoria_pdf_path):,} bytes)")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
