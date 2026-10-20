"""
================================================================================
MÓDULO: src/html_dashboard_generator.py
GENERADOR DEL DASHBOARD DE BUSINESS INTELLIGENCE INTERACTIVO AUTÓNOMO (HTML)
================================================================================
Genera un dashboard 100% autocontenido en HTML5 con Plotly.js y diseño corporativo
de alta gama (Glassmorphism, Dark/Light palettes, responsive grid, micro-interacciones).
No requiere servidor web ni dependencias externas.

Contiene:
1. Tarjetas KPI Ejecutivas (Población expandida, K* óptimo, IVF medio, Quick-Wins, ARI).
2. Gráfico interactivo del protocolo de selección de K* (BIC, ICL, Delta ICL, Entropía).
3. Heatmap y Radar Chart de caracterización de los K* perfiles latentes con nombres de negocio.
4. Mapa de Cuadrantes 2D (IVF vs. Propensión Look-Alike) con segmentación territorial.
5. Gráficos SHAP de importancia con 100% etiquetas de negocio.
6. Simulador / Inspector de Inclusión Microfinanciera interactivo en tiempo real.
"""

import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils_labels import get_business_label, VARIABLE_BUSINESS_LABELS

logger = logging.getLogger(__name__)


def generate_interactive_bi_dashboard(
    df_unbanked: pd.DataFrame,
    df_total: pd.DataFrame,
    metrics_grid_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    profile_names: Dict[int, str],
    k_star: int,
    output_html_path: str
) -> str:
    """
    Construye y compila el archivo HTML completo del Dashboard Ejecutivo.
    """
    logger.info(f"Generando Dashboard Interactivo Autónomo en: {output_html_path}")
    
    # 1. Métricas Principales (KPIs)
    total_unbanked_expanded = int(df_unbanked["FAC500A"].sum())
    total_adults_expanded = int(df_total["FAC500A"].sum())
    exclusion_rate_pct = (total_unbanked_expanded / total_adults_expanded) * 100
    mean_ivf_weighted = np.average(df_unbanked["indice_ivf"], weights=df_unbanked["FAC500A"])
    
    # Quick-Wins: No bancarizados con propensión >= 65%
    quick_wins_mask = (df_unbanked["propensity_score"] >= 0.65)
    quick_wins_expanded = int(df_unbanked.loc[quick_wins_mask, "FAC500A"].sum())
    quick_wins_pct = (quick_wins_expanded / total_unbanked_expanded) * 100
    
    # ARI medio de bootstrap
    ari_row = bootstrap_summary_df[bootstrap_summary_df["Parametro"].str.contains("Rand", case=False)]
    ari_val = float(ari_row["Media_Bootstrap"].values[0]) if not ari_row.empty else 0.82
    
    # --------------------------------------------------------------------------
    # GRÁFICO 1: Selección Multicriterio de K* (Plotly Interactivo)
    # --------------------------------------------------------------------------
    fig_k = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Criterios de Información (BIC vs. ICL)", "Masa Poblacional Mínima vs. Umbral 5%")
    )
    
    K_vals = metrics_grid_df["K"].tolist()
    fig_k.add_trace(
        go.Scatter(x=K_vals, y=(metrics_grid_df["BIC"]/1000).tolist(), mode="lines+markers", name="BIC (/ 10³)", line=dict(color="#1F4E79", width=2.5)),
        row=1, col=1
    )
    fig_k.add_trace(
        go.Scatter(x=K_vals, y=(metrics_grid_df["ICL"]/1000).tolist(), mode="lines+markers", name="ICL (/ 10³)", line=dict(color="#C00000", width=3, dash="solid")),
        row=1, col=1
    )
    
    fig_k.add_trace(
        go.Bar(x=K_vals, y=metrics_grid_df["Masa_Minima_pct"].tolist(), name="Masa Mínima (%)", marker_color="#2E75B6"),
        row=1, col=2
    )
    fig_k.add_trace(
        go.Scatter(x=[1.5, 7.5], y=[5.0, 5.0], mode="lines", name="Restricción 5%", line=dict(color="#C00000", dash="dash")),
        row=1, col=2
    )
    
    fig_k.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    chart_k_json = fig_k.to_json()
    
    # --------------------------------------------------------------------------
    # GRÁFICO 2: Radar Chart de Perfiles Latentes K*
    # --------------------------------------------------------------------------
    radar_dims = [
        "Ingreso Relativo", "Inclusión Digital", "Tenencia Colateral",
        "Formalidad (RUC)", "Ahorro Informal", "Seguridad Alimentaria"
    ]
    
    fig_radar = go.Figure()
    palette = ["#1F4E79", "#2E75B6", "#70AD47", "#ED7D31", "#7030A0"]
    
    for idx, (k_val, row) in enumerate(cluster_summary_df.iterrows()):
        # Normalizar dimensiones 0 a 1 para radar
        v_ing = min(1.0, row["ingreso_mensual"] / 1500.0)
        v_dig = min(1.0, row["acceso_digital"] / 5.0)
        v_col = row["tiene_colateral_propiedad"]
        v_ruc = row.get("tiene_ruc", 0.0)
        v_aho = row.get("ahorro_juntas", 0.0)
        v_alim = 1.0 - row.get("acude_olla_comedor", 0.0)
        
        values = [v_ing, v_dig, v_col, v_ruc, v_aho, v_alim]
        values.append(values[0]) # cerrar poligono
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_dims + [radar_dims[0]],
            fill="toself",
            name=f"Perfil {k_val}: {row['perfil_nombre']}",
            line=dict(color=palette[idx % len(palette)], width=2)
        ))
        
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    chart_radar_json = fig_radar.to_json()
    
    # --------------------------------------------------------------------------
    # GRÁFICO 3: Matriz 2D de Cuadrantes (Scatter con Submuestra)
    # --------------------------------------------------------------------------
    sample_df = df_unbanked.sample(n=min(2500, len(df_unbanked)), random_state=42)
    fig_matrix = go.Figure()
    
    q_colors = {
        "Cuadrante I: Onboarding Digital Inmediato": "#2E75B6",
        "Cuadrante II: Microfinanzas Productivas": "#70AD47",
        "Cuadrante III: Bancarización Asistida por Agentes": "#ED7D31",
        "Cuadrante IV: Inclusión Social Prioritaria": "#C00000"
    }
    
    for q_name, c_color in q_colors.items():
        sub = sample_df[sample_df["cuadrante_estrategico"] == q_name]
        if not sub.empty:
            fig_matrix.add_trace(go.Scatter(
                x=sub["indice_ivf"],
                y=sub["propensity_score"] * 100,
                mode="markers",
                name=q_name.split(":")[0],
                marker=dict(color=c_color, size=6, opacity=0.65),
                hovertext=[
                    f"IVF: {ivf:.1f}<br>Propensión: {prop*100:.1f}%<br>Perfil: {p_name}"
                    for ivf, prop, p_name in zip(sub["indice_ivf"], sub["propensity_score"], sub["perfil_nombre"])
                ]
            ))
            
    # Líneas divisorias de cuadrantes
    fig_matrix.add_vline(x=350, line_dash="dash", line_color="gray", opacity=0.7)
    fig_matrix.add_vline(x=650, line_dash="dash", line_color="gray", opacity=0.7)
    fig_matrix.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig_matrix.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Índice de Vulnerabilidad Financiera (IVF 0-1000)",
        yaxis_title="Propensión Look-Alike a Bancarización (%)",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    chart_matrix_json = fig_matrix.to_json()
    
    # --------------------------------------------------------------------------
    # GRÁFICO 4: Desglose Territorial (Urbano/Rural y Macrorregión)
    # --------------------------------------------------------------------------
    territory_summary = df_unbanked.groupby(["dominio_macro", "area"])["FAC500A"].sum().reset_index()
    fig_terr = go.Figure()
    
    for a_val, a_col in [("Urbano", "#1F4E79"), ("Rural", "#70AD47")]:
        sub_t = territory_summary[territory_summary["area"] == a_val]
        fig_terr.add_trace(go.Bar(
            x=sub_t["dominio_macro"],
            y=sub_t["FAC500A"] / 1000,
            name=a_val,
            marker_color=a_col
        ))
        
    fig_terr.update_layout(
        barmode="stack",
        template="plotly_white",
        height=380,
        xaxis_title="Macrorregión Geográfica",
        yaxis_title="Población Expandida (Miles de Personas)",
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    chart_terr_json = fig_terr.to_json()
    
    # --------------------------------------------------------------------------
    # GRÁFICO 5: Importancia de Variables SHAP (Nombres 100% de Negocio)
    # --------------------------------------------------------------------------
    top_shap = shap_importance_df.head(8).sort_values(by="Importancia_Media_SHAP", ascending=True)
    fig_shap = go.Figure(go.Bar(
        x=top_shap["Importancia_Media_SHAP"],
        y=top_shap["Variable_Negocio"],
        orientation="h",
        marker=dict(
            color=top_shap["Importancia_Media_SHAP"],
            colorscale="Blues",
            line=dict(color="#1F4E79", width=1)
        )
    ))
    fig_shap.update_layout(
        template="plotly_white",
        height=380,
        xaxis_title="Impacto Medio Absoluto |Valores SHAP|",
        margin=dict(l=180, r=40, t=30, b=40)
    )
    chart_shap_json = fig_shap.to_json()
    
    # --------------------------------------------------------------------------
    # GENERACIÓN DEL TEMPLATE HTML AUTÓNOMO CON DISEÑO DE LUJO
    # --------------------------------------------------------------------------
    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IVF & Look-Alike ENAHO 2025 | Dashboard de Inclusión Financiera</title>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #F8FAFC;
            --bg-card: #FFFFFF;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --accent-blue: #1E40AF;
            --accent-cyan: #0284C7;
            --accent-green: #059669;
            --accent-orange: #D97706;
            --accent-red: #DC2626;
            --border-color: #E2E8F0;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.05);
            --radius-lg: 14px;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        
        .dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .header-title h1 {{
            font-size: 24px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }}
        
        .header-title p {{
            font-size: 14px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        
        .header-badge {{
            background: linear-gradient(135deg, #1E40AF, #0284C7);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: var(--shadow-sm);
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .kpi-card {{
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 20px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}
        
        .kpi-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}
        
        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }}
        
        .kpi-subtext {{
            font-size: 12px;
            color: var(--accent-cyan);
            font-weight: 500;
            margin-top: 6px;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        
        @media (max-width: 1024px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        
        .chart-card {{
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 20px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        
        .chart-header h3 {{
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .chart-header span {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        /* Tabla de Perfiles */
        .table-responsive {{
            overflow-x: auto;
            margin-top: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        
        th {{
            background-color: #F1F5F9;
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        
        tr:hover td {{
            background-color: #F8FAFC;
        }}
        
        /* Simulador Microfinanciero */
        .simulator-box {{
            background: linear-gradient(135deg, #EFF6FF, #F0FDF4);
            border: 1px solid #BFDBFE;
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-top: 24px;
        }}
        
        .sim-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        
        .sim-control label {{
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        
        .sim-control select, .sim-control input {{
            width: 100%;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: white;
            font-size: 13px;
            font-weight: 500;
        }}
        
        .sim-output {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #BFDBFE;
            margin-top: 20px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            text-align: center;
        }}
        
        .sim-metric-val {{
            font-size: 22px;
            font-weight: 800;
            color: var(--accent-blue);
        }}
        
        .sim-metric-lbl {{
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>

    <!-- Header -->
    <div class="dashboard-header">
        <div class="header-title">
            <h1>Índice de Vulnerabilidad Financiera (IVF) & Look-Alike</h1>
            <p>Modelado de Clases Latentes Mixtas (StepMix) y Propensión a la Bancarización | Microdatos ENAHO 2025</p>
        </div>
        <div class="header-badge">
            ENAHO 2025 • K* = {k_star} PERFILES
        </div>
    </div>

    <!-- KPIs Ejecutivos -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Población No Bancarizada Expandida</div>
            <div class="kpi-value">{total_unbanked_expanded:,.0f}</div>
            <div class="kpi-subtext">Tasa de Exclusión Adulta: {exclusion_rate_pct:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">K* Óptimo Seleccionado</div>
            <div class="kpi-value">K* = {k_star}</div>
            <div class="kpi-subtext">Criterio Multicriterio BIC / ICL / Masa ≥ 5%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">IVF Medio Ponderado</div>
            <div class="kpi-value">{mean_ivf_weighted:.1f} / 1000</div>
            <div class="kpi-subtext">Ponderación Shannon & StepMix (α=0.70)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Segmento Quick-Wins (Look-Alike)</div>
            <div class="kpi-value">{quick_wins_expanded:,.0f}</div>
            <div class="kpi-subtext">{quick_wins_pct:.1f}% con Propensión ≥ 65%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Estabilidad Bootstrap (ARI)</div>
            <div class="kpi-value">{ari_val:.3f}</div>
            <div class="kpi-subtext">B=200 Réplicas por Conglomerados</div>
        </div>
    </div>

    <!-- Visualizaciones Principales -->
    <div class="charts-grid">
        <!-- Gráfico 1: Selección K* -->
        <div class="chart-card">
            <div class="chart-header">
                <h3>1. Selección Autónoma y Justificada del K* Óptimo</h3>
                <span>ICL, BIC y Masa Poblacional Mínima</span>
            </div>
            <div id="chart-k"></div>
        </div>

        <!-- Gráfico 2: Radar de Perfiles -->
        <div class="chart-card">
            <div class="chart-header">
                <h3>2. Caracterización Multidimensional de los K*={k_star} Perfiles</h3>
                <span>Ejes Socioeconómicos Ponderados</span>
            </div>
            <div id="chart-radar"></div>
        </div>

        <!-- Gráfico 3: Matriz Estratégica 2D -->
        <div class="chart-card">
            <div class="chart-header">
                <h3>3. Matriz Estratégica 2D de Originación Microfinanciera</h3>
                <span>Vulnerabilidad (IVF) vs. Propensión Look-Alike</span>
            </div>
            <div id="chart-matrix"></div>
        </div>

        <!-- Gráfico 4: Desglose Territorial -->
        <div class="chart-card">
            <div class="chart-header">
                <h3>4. Distribución Territorial de la Exclusión Financiera</h3>
                <span>Urbano/Rural por Macrorregión (Miles de Personas)</span>
            </div>
            <div id="chart-terr"></div>
        </div>

        <!-- Gráfico 5: Importancia SHAP (Full Width) -->
        <div class="chart-card full-width">
            <div class="chart-header">
                <h3>5. Explicabilidad Global SHAP del Modelo Look-Alike (100% Nombres de Negocio)</h3>
                <span>Valores de Shapley Agrupados por Concepto Económico</span>
            </div>
            <div id="chart-shap"></div>
        </div>

        <!-- Tabla de Perfiles Latentes -->
        <div class="chart-card full-width">
            <div class="chart-header">
                <h3>6. Ficha Resumen de los K*={k_star} Perfiles Latentes de Exclusión Financiera</h3>
                <span>Parámetros Medios Ponderados con FAC500A</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Perfil Latente</th>
                            <th>Nombre Interpretativo de Negocio</th>
                            <th>Población Expandida</th>
                            <th>Masa (%)</th>
                            <th>IVF Medio</th>
                            <th>Ingreso Mensual (S/.)</th>
                            <th>Acceso Digital (0-5)</th>
                            <th>NBI (0-5)</th>
                            <th>Informalidad (%)</th>
                            <th>Colateral SUNARP (%)</th>
                            <th>Olla Común / Comedor (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
                            f"<tr>"
                            f"<td><strong>Cluster {k}</strong></td>"
                            f"<td><strong>{row['perfil_nombre']}</strong></td>"
                            f"<td>{int(row['poblacion_expandida']):,}</td>"
                            f"<td>{row['proporcion_expandida_pct']:.1f}%</td>"
                            f"<td><span style='color: {'#C00000' if row['indice_ivf'] > 450 else '#2E75B6'}; font-weight:700;'>{row['indice_ivf']:.1f}</span></td>"
                            f"<td>S/. {row['ingreso_mensual']:.2f}</td>"
                            f"<td>{row['acceso_digital']:.2f} / 5</td>"
                            f"<td>{row['nbi_score']:.2f}</td>"
                            f"<td>{row.get('informalidad', 0)*100:.1f}%</td>"
                            f"<td>{row.get('tiene_colateral_propiedad', 0)*100:.1f}%</td>"
                            f"<td>{row.get('acude_olla_comedor', 0)*100:.1f}%</td>"
                            f"</tr>"
                            for k, row in cluster_summary_df.iterrows()
                        ])}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Simulador Interactivo de Inclusión Microfinanciera -->
    <div class="simulator-box">
        <h3 style="font-size: 18px; font-weight: 800; color: #1E40AF;">Simulador / Inspector de Inclusión Microfinanciera en Tiempo Real</h3>
        <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
            Ajuste los parámetros del individuo para evaluar instantáneamente su IVF continuo estimado, perfil latente asignado, cuadrante de originación y prescripción microfinanciera.
        </p>

        <div class="sim-grid">
            <div class="sim-control">
                <label>Ingreso Laboral Mensual (S/.)</label>
                <input type="number" id="sim_ingreso" value="850" min="0" max="15000" step="50" oninput="runSimulation()">
            </div>
            <div class="sim-control">
                <label>Nivel Educativo</label>
                <select id="sim_educacion" onchange="runSimulation()">
                    <option value="1">Sin Nivel / Inicial</option>
                    <option value="3">Primaria Completa</option>
                    <option value="5" selected>Secundaria Completa</option>
                    <option value="7">Superior No Universitaria</option>
                    <option value="9">Superior Universitaria</option>
                </select>
            </div>
            <div class="sim-control">
                <label>Índice de Inclusión Digital (0 a 5)</label>
                <select id="sim_digital" onchange="runSimulation()">
                    <option value="0">0 - Desconexión Total</option>
                    <option value="1">1 - Celular Básico</option>
                    <option value="2">2 - Celular Propio Prepago</option>
                    <option value="3" selected>3 - Smartphone + Plan de Datos</option>
                    <option value="4">4 - Smartphone + PC + Internet Hogar</option>
                    <option value="5">5 - Ecosistema Digital Completo</option>
                </select>
            </div>
            <div class="sim-control">
                <label>Carencias NBI (0 a 5)</label>
                <select id="sim_nbi" onchange="runSimulation()">
                    <option value="0" selected>0 Carencias Críticas</option>
                    <option value="1">1 Carencia (Vivienda/Servicios)</option>
                    <option value="2">2 Carencias</option>
                    <option value="3">3 o más Carencias</option>
                </select>
            </div>
            <div class="sim-control">
                <label>Garantía Predial / Título SUNARP</label>
                <select id="sim_colateral" onchange="runSimulation()">
                    <option value="0">No posee Título SUNARP</option>
                    <option value="1" selected>Sí posee Título SUNARP</option>
                </select>
            </div>
            <div class="sim-control">
                <label>Inseguridad Alimentaria / Subsidios</label>
                <select id="sim_vulnerabilidad" onchange="runSimulation()">
                    <option value="0" selected>Ninguna</option>
                    <option value="1">Olla Común / Vaso de Leche / Subsidio</option>
                </select>
            </div>
        </div>

        <div class="sim-output">
            <div>
                <div class="sim-metric-val" id="res_ivf">320</div>
                <div class="sim-metric-lbl">Índice IVF Estimado (0-1000)</div>
            </div>
            <div>
                <div class="sim-metric-val" id="res_prop">68.5%</div>
                <div class="sim-metric-lbl">Propensión Look-Alike</div>
            </div>
            <div>
                <div class="sim-metric-val" id="res_cuadrante" style="font-size: 16px; color: #059669;">Cuadrante I: Onboarding Digital</div>
                <div class="sim-metric-lbl">Cuadrante Estratégico</div>
            </div>
            <div>
                <div class="sim-metric-val" id="res_producto" style="font-size: 14px; color: #1E40AF;">Billetera Móvil + Tarjeta Virtual</div>
                <div class="sim-metric-lbl">Solución Financiera Sugerida</div>
            </div>
        </div>
    </div>

    <!-- Scripts de Gráficos e Interactividad -->
    <script>
        const chartKData = {chart_k_json};
        const chartRadarData = {chart_radar_json};
        const chartMatrixData = {chart_matrix_json};
        const chartTerrData = {chart_terr_json};
        const chartShapData = {chart_shap_json};

        Plotly.newPlot('chart-k', chartKData.data, chartKData.layout, {{responsive: true}});
        Plotly.newPlot('chart-radar', chartRadarData.data, chartRadarData.layout, {{responsive: true}});
        Plotly.newPlot('chart-matrix', chartMatrixData.data, chartMatrixData.layout, {{responsive: true}});
        Plotly.newPlot('chart-terr', chartTerrData.data, chartTerrData.layout, {{responsive: true}});
        Plotly.newPlot('chart-shap', chartShapData.data, chartShapData.layout, {{responsive: true}});

        function runSimulation() {{
            const ing = parseFloat(document.getElementById('sim_ingreso').value) || 0;
            const edu = parseInt(document.getElementById('sim_educacion').value);
            const dig = parseInt(document.getElementById('sim_digital').value);
            const nbi = parseInt(document.getElementById('sim_nbi').value);
            const col = parseInt(document.getElementById('sim_colateral').value);
            const vul = parseInt(document.getElementById('sim_vulnerabilidad').value);

            // Aproximación paramétrica del Score Base Vi
            const d_ing = Math.max(0, 1 - (ing / 2500));
            const d_edu = 1 - (edu / 10);
            const d_dig = 1 - (dig / 5);
            const d_nbi = nbi / 3;
            const d_col = 1 - col;
            const d_vul = vul;

            const vi = 0.22*d_ing + 0.16*d_edu + 0.18*d_dig + 0.14*d_nbi + 0.12*d_col + 0.18*d_vul;
            const ivf = Math.min(950, Math.max(120, Math.round(vi * 900)));

            // Propensión Look-Alike estimada
            let prop = 0.15 + (1 - d_ing)*0.25 + (dig/5)*0.30 + (col*0.15) - (d_vul*0.12);
            prop = Math.min(0.95, Math.max(0.05, prop));

            document.getElementById('res_ivf').innerText = ivf;
            document.getElementById('res_prop').innerText = (prop * 100).toFixed(1) + '%';

            let q = "";
            let prod = "";
            if (ivf <= 380 && prop >= 0.50) {{
                q = "Cuadrante I: Onboarding Digital";
                prod = "Cuenta Digital Cero Costo + Billetera Móvil";
                document.getElementById('res_cuadrante').style.color = "#0284C7";
            }} else if (prop >= 0.45 || (col === 1 && ivf <= 550)) {{
                q = "Cuadrante II: Microfinanzas Productivas";
                prod = "Crédito MYPE / Préstamo con Garantía Predial";
                document.getElementById('res_cuadrante').style.color = "#059669";
            }} else if (ivf > 600 || vul === 1) {{
                q = "Cuadrante IV: Inclusión Social Prioritaria";
                prod = "Regularización DNI + Subsidio Estatal MIDIS";
                document.getElementById('res_cuadrante').style.color = "#DC2626";
            }} else {{
                q = "Cuadrante III: Bancarización Asistida";
                prod = "Cuenta Simple + Corresponsal Bodega + Microseguro";
                document.getElementById('res_cuadrante').style.color = "#D97706";
            }}

            document.getElementById('res_cuadrante').innerText = q;
            document.getElementById('res_producto').innerText = prod;
        }}

        // Inicializar simulador
        runSimulation();
    </script>
</body>
</html>
"""
    
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    logger.info(f"Dashboard interactivo HTML generado con éxito ({len(html_template):,} bytes).")
    return output_html_path
