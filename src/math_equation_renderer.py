"""
================================================================================
MÓDULO: src/math_equation_renderer.py
RENDERIZADOR VECTORIAL DE ECUACIONES MATEMÁTICAS EN ALTA RESOLUCIÓN (300 DPI)
================================================================================
Genera tarjetas gráficas profesionales con fórmulas en LaTeX renderizadas mediante
el motor tipográfico de Matplotlib a 300 DPI, con fondo estilizado (#F8FAFC),
bordes redondeados y numeración formal de ecuaciones. Esto resuelve la incapacidad
nativa de ReportLab para renderizar notación matemática compleja, asegurando que
los informes en PDF tengan calidad editorial científica.
"""

import os
import logging
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)

EQUATION_DEFINITIONS: Dict[str, Tuple[str, str, float]] = {
    "eq1_lca": (
        r'P(X_i = x_i \mid \Theta) = \sum_{k=1}^K \pi_k \left[ \prod_{j=1}^{M_{\mathrm{cat}}} \prod_{c=1}^{C_j} (\lambda_{kjc})^{\mathbf{1}(x_{ij}=c)} \right] \cdot \frac{\exp\left(-\frac{1}{2}(x_i^{\mathrm{cont}} - \mu_k)^T \Sigma_k^{-1}(x_i^{\mathrm{cont}} - \mu_k)\right)}{(2\pi)^{M_{\mathrm{cont}}/2} |\Sigma_k|^{1/2}}',
        "1",
        9.2
    ),
    "eq2_icl": (
        r'\mathrm{ICL}(K) = \mathrm{BIC}(K) + 2 \, E_w(K) = \left[ -2 \ln L_w(K) + d_K \ln N \right] - 2 \sum_{i=1}^N w_i \sum_{k=1}^K \gamma_{ik} \ln(\gamma_{ik} + \epsilon)',
        "2",
        9.5
    ),
    "eq3_delta_icl": (
        r'\Delta \mathrm{ICL}_K = \frac{\mathrm{ICL}(K-1) - \mathrm{ICL}(K)}{\mathrm{ICL}(K-1)} \times 100\%, \qquad E_{\mathrm{rel}}(K) = \frac{-\sum_{i=1}^N w_i \sum_{k=1}^K \gamma_{ik} \ln(\gamma_{ik} + \epsilon)}{N_w \ln K} < 0.15',
        "3",
        8.8
    ),
    "eq4_shannon": (
        r'e_j = -\frac{1}{\ln N} \sum_{i=1}^N p_{ij} \ln(p_{ij} + \epsilon), \qquad d_j = 1 - e_j, \qquad w_j = \frac{d_j}{\sum_{m=1}^M d_m}',
        "4",
        9.8
    ),
    "eq5_rk": (
        r'R_k = \frac{\sum_{i=1}^N w_i \, \gamma_{ik} \, V_i}{\sum_{i=1}^N w_i \, \gamma_{ik}}, \qquad \gamma_{ik} = P(Z_i = k \mid X_i = x_i)',
        "5",
        10.0
    ),
    "eq6_ivf": (
        r'\mathrm{IVF}_i = 1000 \times \sum_{k=1}^{K^*} \gamma_{ik} \left[ \alpha \, R_k + (1 - \alpha) \, V_i \right], \qquad \alpha = 0.70',
        "6",
        10.2
    ),
    "eq7_isotonic": (
        r'\hat{p}_{\mathrm{cal}}(x) = \arg\min_{\hat{m}} \sum_{i=1}^N w_i \left( y_i - \hat{m}(\hat{s}_i) \right)^2 \quad \mathrm{s.a.} \quad \hat{m}(\hat{s}_{(1)}) \leq \dots \leq \hat{m}(\hat{s}_{(N)})',
        "7",
        9.2
    ),
    "eq8_shap": (
        r'\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]',
        "8",
        10.0
    ),
    "eq9_bootstrap": (
        r'\hat{\theta}_{HT}^{(b)} = \frac{\sum_{i \in \mathrm{S}^{(b)}} w_i^{(b)} Y_i}{\sum_{i \in \mathrm{S}^{(b)}} w_i^{(b)}}, \qquad \sigma^* = \arg\max_{\pi \in \mathrm{P}_K} \sum_{k=1}^K \mathrm{S}_{k, \pi(k)}',
        "9",
        9.8
    )
}


def render_all_equations(output_dir: str = "reports/figures/equations", dpi: int = 300) -> Dict[str, str]:
    """
    Renderiza todas las fórmulas matemáticas en archivos PNG de alta resolución.
    Retorna un diccionario con las rutas de cada imagen generada.
    """
    os.makedirs(output_dir, exist_ok=True)
    rendered_paths: Dict[str, str] = {}
    
    for eq_key, (latex_code, eq_number, font_sz) in EQUATION_DEFINITIONS.items():
        filepath = os.path.join(output_dir, f"{eq_key}.png")
        
        # Dimensiones optimizadas para ancho completo en página Letter
        fig = plt.figure(figsize=(7.2, 0.85), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        
        # Tarjeta con fondo tenue y borde suave
        box = FancyBboxPatch(
            (0.005, 0.05), 0.99, 0.90,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=0.8,
            transform=ax.transAxes
        )
        ax.add_patch(box)
        
        ax.text(
            0.48, 0.50, f"${latex_code}$",
            fontsize=font_sz, ha="center", va="center",
            color="#0F172A", transform=ax.transAxes
        )
        
        # Número de ecuación a la derecha
        if eq_number:
            ax.text(
                0.97, 0.50, f"({eq_number})",
                fontsize=font_sz - 0.5, ha="right", va="center",
                color="#64748B", fontweight="bold", transform=ax.transAxes
            )
            
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        rendered_paths[eq_key] = filepath
        
    logger.info(f"Se renderizaron {len(rendered_paths)} fórmulas matemáticas vectoriales en {output_dir}")
    return rendered_paths
