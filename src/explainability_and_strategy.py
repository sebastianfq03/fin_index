"""
================================================================================
MÓDULO: src/explainability_and_strategy.py
ETAPA 5: EXPLICABILIDAD SHAP, MATRIZ ESTRATÉGICA 2D Y GENERACIÓN DE FIGURAS
================================================================================
Fundamentación Teórica y Econométrica:
--------------------------------------------------------------------------------
1. Teoría de Juegos Cooperativos y Valores de Shapley (SHAP):
   Para superar la opacidad de los algoritmos de ensamble no lineales y garantizar
   la explicabilidad de la propensión a la bancarización y la vulnerabilidad,
   se aplica la formulación axiomática de Shapley (1953) y TreeSHAP (Lundberg et al., 2020):
       \\phi_j(x) = \\sum_{S \\subseteq F \\setminus \\{j\\}} \\frac{|S|! (|F| - |S| - 1)!}{|F|!} \\left[ f_x(S \\cup \\{j\\}) - f_x(S) \\right]
   Propiedades matemáticas garantizadas:
   - Eficiencia: La suma de los valores de Shapley reproduce exactamente la desviación
     de la predicción respecto a la tasa base poblacional: \\sum_j \\phi_j = f(x) - E[f(x)].
   - Simetría: Si dos variables contribuyen de igual manera a cualquier coalición,
     sus valores de Shapley son estrictamente idénticos.
   - Jugador Nulo: Variables irrelevantes reciben un valor de Shapley exactamente cero.
   - Aditividad: Permite agregar dummies de variables politómicas sin alterar la consistencia.

2. Matriz Estratégica 2D de Inclusión Financiera (IVF vs. Propensión Look-Alike):
   Segmenta el mercado no bancarizado cruzando la vulnerabilidad multidimensional (eje X)
   con la propensión estadística de adopción (eje Y) en 4 cuadrantes prescriptivos:
   - Cuadrante I: Onboarding Digital Inmediato (Bajo IVF, Alta Propensión).
   - Cuadrante II: Microfinanzas Productivas (IVF Medio, Propensión Media/Alta).
   - Cuadrante III: Bancarización Asistida por Agentes (Alto IVF, Baja Propensión).
   - Cuadrante IV: Inclusión Social Prioritaria (IVF Crítico, Baja Propensión).
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.utils_labels import get_business_label, VARIABLE_BUSINESS_LABELS

logger = logging.getLogger(__name__)

# Configuración de estética corporativa y académica para figuras
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


def assign_strategic_quadrants(
    df_unbanked: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Clasifica a cada individuo no bancarizado en los 4 cuadrantes estratégicos de
    originación microfinanciera en función de su IVF (0-1000) y su Propensity Score (0-1).
    """
    logger.info("Asignando Cuadrantes de Inclusión Microfinanciera 2D...")
    df = df_unbanked.copy()
    
    ivf_low = config["strategic_matrix"]["ivf_thresholds"]["low"]
    ivf_high = config["strategic_matrix"]["ivf_thresholds"]["high"]
    prop_low = config["strategic_matrix"]["propensity_thresholds"]["low"]
    prop_high = config["strategic_matrix"]["propensity_thresholds"]["high"]
    
    # Condiciones
    ivf = df["indice_ivf"].values
    prop = df["propensity_score"].values
    
    cuadrantes = []
    estrategias = []
    productos = []
    canales = []
    
    for i in range(len(df)):
        ivf_i = ivf[i]
        prop_i = prop[i]
        
        # Cuadrante I: Bajo IVF (<350) + Alta/Media Propensión (>=0.45)
        if ivf_i <= ivf_high and prop_i >= prop_low:
            if prop_i >= prop_high and ivf_i <= ivf_low:
                q = "Cuadrante I: Onboarding Digital Inmediato"
                e = "Inclusión Financiera Digital Autogestionada (Quick-Wins)"
                p = "Cuenta Digital Cero Costo + Billetera Móvil + Tarjeta Virtual"
                c = "App Móvil / Billetera Digital (Yape/Plin/Banca Móvil)"
            else:
                q = "Cuadrante II: Microfinanzas Productivas"
                e = "Financiamiento MYPE y Bancarización con Garantía"
                p = "Crédito Capital de Trabajo MYPE / Préstamo con Garantía Predial"
                c = "Agencia Express / Oficial de Microfinanzas en Terreno"
        elif ivf_i > ivf_high and prop_i < prop_high:
            if ivf_i >= 550 or df.iloc[i].get("acude_olla_comedor", 0) == 1 or df.iloc[i].get("sin_dni_barrera", 0) == 1:
                q = "Cuadrante IV: Inclusión Social Prioritaria"
                e = "Protección Social, Identidad KYC y Subsidios de Alivio"
                p = "Regularización de DNI + Canalización de Transferencias (JUNTOS/Pensión 65)"
                c = "Módulos Itinerantes MIDIS/RENIEC / Banco de la Nación"
            else:
                q = "Cuadrante III: Bancarización Asistida por Agentes"
                e = "Educación Financiera Básica y Acompañamiento Físico"
                p = "Cuenta Básica de Ahorros + Microseguro de Salud + Microahorro Programado"
                c = "Agentes Corresponsales No Bancarios (Bodegas / Farmacias)"
        else:
            # Casos intermedios asignados por nivel de propensión dominante
            if prop_i >= 0.50:
                q = "Cuadrante II: Microfinanzas Productivas"
                e = "Fortalecimiento Productivo y Digitalización de Flujos"
                p = "Línea de Crédito Productivo + Cobro con QR"
                c = "Oficial de Crédito MYPE / Corresponsal"
            else:
                q = "Cuadrante III: Bancarización Asistida por Agentes"
                e = "Acceso Gradual con Microahorro Protegido"
                p = "Cuenta Simple + Microseguro Familiar"
                c = "Agente Corresponsal / Ventanilla"
                
        cuadrantes.append(q)
        estrategias.append(e)
        productos.append(p)
        canales.append(c)
        
    df["cuadrante_estrategico"] = cuadrantes
    df["estrategia_recomendada"] = estrategias
    df["producto_recomendado"] = productos
    df["canal_prioritario"] = canales
    
    # Resumen cuantitativo ponderado
    weight_col = config["data_specs"]["weight_col"]
    weights = df[weight_col].values
    total_w = weights.sum()
    
    logger.info("=== DISTRIBUCIÓN POBLACIONAL EXPANDIDA POR CUADRANTE ESTRATÉGICO ===")
    for q_name in sorted(df["cuadrante_estrategico"].unique()):
        mask = (df["cuadrante_estrategico"] == q_name)
        w_sub = weights[mask].sum()
        pct = (w_sub / total_w) * 100
        ivf_m = np.average(df.loc[mask, "indice_ivf"], weights=weights[mask])
        prop_m = np.average(df.loc[mask, "propensity_score"], weights=weights[mask])
        logger.info(f"{q_name} | Población: {int(w_sub):,} ({pct:.1f}%) | IVF Prom: {ivf_m:.1f} | Propensión: {prop_m*100:.1f}%")
        
    return df


def compute_shap_explanations(
    base_model: Any,
    df_sample: pd.DataFrame,
    features: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calcula los Valores de Shapley (SHAP) sobre el modelo Look-Alike CatBoost,
    traduce el 100% de las variables a lenguaje de negocio y calcula importancia relativa.
    """
    # Asegurar que X_sample tenga la misma estructura de columnas codificadas
    if all(col in df_sample.columns for col in features):
        X_sample = df_sample[features].copy()
    else:
        from src.lookalike_propensity_model import prepare_lookalike_matrices
        X_sample, _ = prepare_lookalike_matrices(df_sample, feature_cols=features)
        
    # Submuestra representativa si es mayor a 2,500 para eficiencia computacional
    if len(X_sample) > 2500:
        X_sample = X_sample.sample(n=2500, random_state=42)
        
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Importancia media absoluta de Shapley sum(|phi_ij|) / N
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Mapeo a nombres de negocio
    business_names = [get_business_label(col) for col in features]
    
    shap_importance_df = pd.DataFrame({
        "Variable_Original": features,
        "Variable_Negocio": business_names,
        "Importancia_Media_SHAP": mean_abs_shap
    }).sort_values(by="Importancia_Media_SHAP", ascending=False)
    
    # Porcentaje de impacto relativo
    shap_importance_df["Impacto_Relativo_pct"] = (
        (shap_importance_df["Importancia_Media_SHAP"] / shap_importance_df["Importancia_Media_SHAP"].sum()) * 100
    ).round(2)
    
    return shap_values, shap_importance_df


def generate_research_figures(
    df_unbanked: pd.DataFrame,
    metrics_grid_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    bootstrap_data: Dict[str, Any],
    cluster_summary_df: pd.DataFrame,
    figures_dir: str
) -> Dict[str, str]:
    """
    Genera y guarda en reports/figures/ todos los gráficos estáticos de alta resolución
    (300 DPI) requeridos para el Informe Científico en PDF:
    1. fig_k_selection.png
    2. fig_ivf_distribution.png
    3. fig_strategic_matrix_2d.png
    4. fig_shap_importance.png
    5. fig_bootstrap_validation.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    generated_paths = {}
    
    # --------------------------------------------------------------------------
    # FIGURA 1: Protocolo Multicriterio de Selección de K* (BIC, ICL y Entropía)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    K_vals = metrics_grid_df["K"]
    ax1.plot(K_vals, metrics_grid_df["BIC"] / 1000, "o--", color="#1F4E79", label="BIC (/ 10³)", linewidth=1.8)
    ax1.plot(K_vals, metrics_grid_df["ICL"] / 1000, "s-", color="#C00000", label="ICL (/ 10³)", linewidth=2.2)
    ax1.set_xlabel("Número de Clases Latentes (K)", fontweight="bold")
    ax1.set_ylabel("Criterio de Información (× 1,000)", fontweight="bold")
    ax1.set_title("Ajuste y Penalización de Clasificación (BIC vs. ICL)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)
    
    ax2.plot(K_vals, metrics_grid_df["Masa_Minima_pct"], "d-", color="#2E75B6", label="Masa Poblacional Mínima (%)", linewidth=2.0)
    ax2.axhline(5.0, color="#C00000", linestyle="--", linewidth=1.5, label="Restricción Mínima (5%)")
    ax2.set_xlabel("Número de Clases Latentes (K)", fontweight="bold")
    ax2.set_ylabel("Masa Poblacional Mínima (%)", fontweight="bold")
    ax2.set_title("Representatividad y Criterio de Masa Mínima (5%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    p1 = os.path.join(figures_dir, "fig_k_selection.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    generated_paths["k_selection"] = p1
    
    # --------------------------------------------------------------------------
    # FIGURA 2: Distribución Continua del IVF por Perfil Latente K*
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = ["#2E75B6", "#ED7D31", "#70AD47", "#7030A0", "#FFC000"]
    
    for idx, (k_val, row) in enumerate(cluster_summary_df.iterrows()):
        sub_df = df_unbanked[df_unbanked["cluster_lca"] == k_val]
        color = palette[idx % len(palette)]
        name = row["perfil_nombre"]
        sns.kdeplot(
            sub_df["indice_ivf"],
            ax=ax,
            color=color,
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            label=f"{name} (Medio: {row['indice_ivf']:.0f})"
        )
        
    ax.set_title("Distribución Continua del Índice de Vulnerabilidad Financiera (IVF) por Perfil", fontsize=11, fontweight="bold")
    ax.set_xlabel("Índice de Vulnerabilidad Financiera (Escala 0 a 1000 Puntos)", fontweight="bold")
    ax.set_ylabel("Densidad Empírica Ponderada", fontweight="bold")
    ax.set_xlim(0, 1000)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    p2 = os.path.join(figures_dir, "fig_ivf_distribution.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    generated_paths["ivf_distribution"] = p2
    
    # --------------------------------------------------------------------------
    # FIGURA 3: Matriz Estratégica 2D (IVF vs. Propensión a Bancarización)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.0))
    sample_vis = df_unbanked.sample(n=min(3000, len(df_unbanked)), random_state=42)
    
    q_colors = {
        "Cuadrante I: Onboarding Digital Inmediato": "#2E75B6",
        "Cuadrante II: Microfinanzas Productivas": "#70AD47",
        "Cuadrante III: Bancarización Asistida por Agentes": "#ED7D31",
        "Cuadrante IV: Inclusión Social Prioritaria": "#C00000"
    }
    
    for q_name, c_color in q_colors.items():
        q_sub = sample_vis[sample_vis["cuadrante_estrategico"] == q_name]
        if not q_sub.empty:
            ax.scatter(
                q_sub["indice_ivf"],
                q_sub["propensity_score"] * 100,
                color=c_color,
                alpha=0.6,
                s=20,
                label=q_name.split(":")[1].strip()
            )
            
    ax.axvline(350, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(650, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Índice de Vulnerabilidad Financiera (IVF 0-1000)", fontweight="bold")
    ax.set_ylabel("Propensión a Bancarización Look-Alike (%)", fontweight="bold")
    ax.set_title("Matriz de Originación Microfinanciera 2D (Vulnerabilidad vs. Propensión)", fontsize=11, fontweight="bold")
    ax.set_xlim(100, 900)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    p3 = os.path.join(figures_dir, "fig_strategic_matrix_2d.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    generated_paths["strategic_matrix"] = p3
    
    # --------------------------------------------------------------------------
    # FIGURA 4: Importancia de Variables SHAP (100% Nombres de Negocio)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    top_shap = shap_importance_df.head(10).sort_values(by="Importancia_Media_SHAP", ascending=True)
    
    bars = ax.barh(top_shap["Variable_Negocio"], top_shap["Importancia_Media_SHAP"], color="#1F4E79", height=0.65)
    ax.set_xlabel("Impacto Medio Absoluto en la Predicción |Valores SHAP|", fontweight="bold")
    ax.set_title("Top 10 Determinantes de la Bancarización (Valores de Shapley Agrupados)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=8.5, color="#1F4E79")
        
    plt.tight_layout()
    p4 = os.path.join(figures_dir, "fig_shap_importance.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    generated_paths["shap_importance"] = p4
    
    # --------------------------------------------------------------------------
    # FIGURA 5: Distribución Bootstrap del IVF y Estabilidad ARI
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    sns.histplot(bootstrap_data["b_mean_ivf"], kde=True, ax=ax1, color="#2E75B6", bins=18)
    ax1.set_title("Distribución Bootstrap del IVF Medio (B=200)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("IVF Medio Poblacional", fontweight="bold")
    ax1.set_ylabel("Frecuencia de Réplicas", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    sns.histplot(bootstrap_data["b_ari_scores"], kde=True, ax=ax2, color="#70AD47", bins=18)
    ax2.set_title("Estabilidad de Partición (Adjusted Rand Index)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Índice de Rand Ajustado (ARI)", fontweight="bold")
    ax2.set_ylabel("Frecuencia de Réplicas", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    p5 = os.path.join(figures_dir, "fig_bootstrap_validation.png")
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    generated_paths["bootstrap_validation"] = p5
    
    logger.info(f"5 Figuras de investigación generadas en {figures_dir} con calidad de imprenta (300 DPI).")
    return generated_paths
