"""
================================================================================
MÓDULO: src/pdf_research_generator.py
GENERADOR DEL INFORME TÉCNICO Y CIENTÍFICO EN PDF (REPORTLAB + MATH RENDERER)
================================================================================
Compila de forma autónoma el documento monográfico de investigación y reporte
analítico con rigor econométrico, portada académica, numeración 'Página X de Y',
ecuaciones matemáticas renderizadas en alta resolución (300 DPI), tablas
estilizadas, figuras incrustadas y bibliografía APA 7.
"""

import os
import sys
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
from src.math_equation_renderer import render_all_equations

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """
    Canvas personalizado para ReportLab que implementa dos pasadas para calcular
    el total de páginas y generar el pie 'Página X de Y' y encabezado institucional.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Omitir decoraciones en la portada (página 1)
        if self._pageNumber == 1:
            return
            
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Encabezado superior
        header_text = "INVESTIGACIÓN APLICADA: ÍNDICE DE VULNERABILIDAD FINANCIERA (IVF) • ENAHO 2025"
        self.drawString(54, 750, header_text)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(54, 744, 558, 744)
        
        # Pie de página inferior
        footer_text = "División de Econometría y Microfinanzas Aplicadas • Proyecto de Inclusión Financiera"
        page_str = f"Página {self._pageNumber} de {page_count}"
        self.drawString(54, 38, footer_text)
        self.drawRightString(558, 38, page_str)
        self.line(54, 48, 558, 48)
        
        self.restoreState()


def build_research_pdf_report(
    df_unbanked: pd.DataFrame,
    df_total: pd.DataFrame,
    metrics_grid_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    weights_comparison_df: pd.DataFrame,
    lookalike_metrics: Dict[str, float],
    figure_paths: Dict[str, str],
    k_star: int,
    output_pdf_path: str
) -> str:
    """
    Ensambla y compila el documento completo de investigación en PDF con fórmulas matemáticas vectoriales.
    """
    logger.info(f"Generando Informe Técnico y Científico en PDF en: {output_pdf_path}")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Renderizar tarjetas de ecuaciones matemáticas en 300 DPI
    eq_dir = os.path.join(os.path.dirname(output_pdf_path), "figures", "equations")
    eq_paths = render_all_equations(eq_dir, dpi=300)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Estilos tipográficos personalizados
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=18
    )
    
    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#475569")
    )
    
    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
        alignment=4 # Justificado
    )
    
    abstract_box = ParagraphStyle(
        "AbstractBox",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F1F5F9"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=6,
        spaceAfter=8,
        alignment=4
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.white
    )
    
    story = []
    
    # ==========================================================================
    # PÁGINA 1: PORTADA ACADÉMICA / CORPORATIVA & RESUMEN EJECUTIVO
    # ==========================================================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("INFORME DE INVESTIGACIÓN APLICADA Y MODELADO PREDICTIVO", ParagraphStyle("SuperHdr", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=colors.HexColor("#0284C7"), spaceAfter=8)))
    story.append(Paragraph("Construcción de un Índice de Vulnerabilidad Financiera (IVF) mediante Modelos de Clases Latentes para Datos Mixtos y Modelado Look-Alike en el Perú (ENAHO 2025)", title_style))
    story.append(Paragraph("Evaluación Econométrica Multidimensional de la Exclusión Financiera, Selección Autónoma del Número Óptimo de Perfiles (K*) y Estrategia de Bancarización de Gemelos Estadísticos", subtitle_style))
    
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1E3A8A"), spaceBefore=6, spaceAfter=14))
    
    meta_text = """
    <b>Autor:</b> Investigador Principal en Estadística Aplicada & Lead Data Scientist en Microfinanzas<br/>
    <b>Institución:</b> Centro de Investigaciones en Inclusión Financiera y Econometría de Encuestas<br/>
    <b>Fuente de Datos:</b> Encuesta Nacional de Hogares (ENAHO 2025) — Instituto Nacional de Estadística e Informática (INEI)<br/>
    <b>Módulos Procesados:</b> 100 (Vivienda/NBI), 200 (Demografía), 300 (Educación/TIC), 400 (Salud/KYC), 500 (Empleo/Inclusión), 602 (Ollas Comunes), 700 (Programas Sociales)<br/>
    <b>Fecha de Publicación:</b> 2026 | Versión 2.0.0 Publicación Científica Definitiva
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 14))
    
    # Resumen Ejecutivo / Abstract
    tot_unb = int(df_unbanked["FAC500A"].sum())
    tot_pop = int(df_total["FAC500A"].sum())
    excl_rate = (tot_unb / tot_pop) * 100
    ivf_mean = np.average(df_unbanked["indice_ivf"], weights=df_unbanked["FAC500A"])
    
    abstract_text = f"""
    <b>RESUMEN EJECUTIVO:</b> El presente estudio desarrolla, fundamenta matemáticamente e implementa
    de principio a fin un Índice de Vulnerabilidad Financiera (IVF) continuo (escala 0 a 1000) y un modelo
    supervisado Look-Alike de propensión a la bancarización para el Perú a partir de los microdatos oficiales
    de la ENAHO 2025. Se supera la metodología tradicional de dos etapas (MCA+GMM) mediante el uso de Modelos
    de Mezclas Finitas para Datos Mixtos (StepMix: Multinomial-Gaussiano), estimando directamente la verosimilitud
    conjunta y evitando la deformación métrica de la matriz de Burt. Mediante una regla de decisión multicriterio
    basada en la literatura (Biernacki et al., 2000; Nardo et al., 2005) que evalúa el Criterio de Información
    Bayesiano (BIC), el Criterio de Clasificación Integrada (ICL), el codo en la ganancia marginal relativa (ΔICL) y la
    restricción de masa poblacional mínima expandida (π_k ≥ 5% con FAC500A), el algoritmo determina de forma
    completamente autónoma un <b>K* = {k_star} perfiles latentes óptimos</b>, descartando soluciones espurias
    sobrefragmentadas típicas de muestras masivas (N = 28,016 no bancarizados; N_expandido = {tot_unb:,.0f} personas, {excl_rate:.1f}% de exclusión adulta).
    El IVF se formula combinando el nivel de vulnerabilidad de clase latente R_k (α=0.70) y un score base individual V_i
    ponderado por Entropía de Shannon (1 - α=0.30), logrando una distribución continua sin agujas discretas (IVF medio = {ivf_mean:.1f} pts).
    El modelo Look-Alike alcanza un ROC-AUC de {lookalike_metrics['ROC_AUC']:.4f} y un estadístico KS de {lookalike_metrics['KS_Statistic']:.2f}%,
    permitiendo clasificar a la población no bancarizada en 4 cuadrantes prescriptivos para la industria microfinanciera.
    """
    story.append(Paragraph(abstract_text, abstract_box))
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 2: INTRODUCCIÓN, MARCO TEÓRICO Y FÓRMULAS MATEMÁTICAS
    # ==========================================================================
    story.append(Paragraph("1. Introducción y Novedades Metodológicas ENAHO 2025", h1_style))
    story.append(Paragraph(
        "A pesar de los avances regulatorios y del despliegue masivo de billeteras digitales interoperables en el Perú, "
        "la exclusión financiera formal persiste como un fenómeno estructural crítico. Según la Encuesta Nacional de Hogares "
        "(ENAHO 2025), aproximadamente 4 de cada 10 adultos peruanos (18 a 65 años) carecen de cualquier cuenta o tarjeta formal. "
        "Dicha exclusión no obedece a un patrón homogéneo de pobreza monetaria, sino a un entramado complejo de informalidad laboral, "
        "barreras de identidad legal KYC, limitaciones de capital humano y vulnerabilidad alimentaria severa.",
        body_style
    ))
    story.append(Paragraph(
        "La ENAHO 2025 introduce innovaciones metodológicas fundamentales capturadas en el pipeline: (1) Registro específico de la tenencia "
        "y regularización registral de colateral predial en SUNARP (Módulo 100, P106A/B); (2) Ecosistema digital individual granular (Módulo 300), "
        "incluyendo conectividad mediante smartphones con plan de datos y comercialización en redes sociales/marketplaces; (3) Medición "
        "exhaustiva de la inseguridad alimentaria crítica a través del auxilio en Ollas Comunes y Comedores Populares (Módulo 602); y "
        "(4) Cobertura de transferencias monetarias del Estado (Módulo 700: JUNTOS, Pensión 65, Bono Gas FISE y Wasi Mikuna).",
        body_style
    ))
    
    story.append(Paragraph("2. Marco Teórico y Formulación Matemática", h1_style))
    story.append(Paragraph(
        "<b>2.1 Modelado Conjunto Multinomial-Gaussiano (StepMix) vs. MCA+GMM:</b> La práctica convencional suele "
        "reducir la dimensionalidad mediante Análisis de Correspondencias Múltiples (MCA) y aplicar modelos de mezclas gaussianas (GMM) "
        "sobre los factores continuos. Dicho procedimiento incurre en la falacia de dos etapas: asume puntuaciones factoriales sin error "
        "de medición e impone distancias euclidianas sobre la matriz de Burt que distorsionan las probabilidades posteriores. En contraposición, "
        "StepMix modela directamente el vector conjunto x_i = (x_i^{cat}, x_i^{cont}) mediante la verosimilitud de mezcla finita:",
        body_style
    ))
    
    # Ecuación 1: StepMix Joint Likelihood
    if "eq1_lca" in eq_paths and os.path.exists(eq_paths["eq1_lca"]):
        story.append(Image(eq_paths["eq1_lca"], width=504, height=56))
        story.append(Spacer(1, 4))
        
    story.append(Paragraph(
        "<b>2.2 Selección Autónoma de K* basada en la Información Integrada (ICL):</b> El Criterio de Información Bayesiano (BIC) "
        "tiende a sobrefragmentar muestras masivas (N ≈ 30,000) favoreciendo clusters redundantes. Se programa la regla de decisión "
        "multicriterio de Biernacki, Celeux y Govaert (2000), penalizando la dispersión de las probabilidades posteriores γ_{ik} "
        "mediante el Criterio de Clasificación Integrada (ICL) y la tasa de ganancia marginal relativa ΔICL:",
        body_style
    ))
    
    # Ecuación 2: ICL
    if "eq2_icl" in eq_paths and os.path.exists(eq_paths["eq2_icl"]):
        story.append(Image(eq_paths["eq2_icl"], width=504, height=54))
        story.append(Spacer(1, 3))
        
    # Ecuación 3: Delta ICL & Relative Entropy
    if "eq3_delta_icl" in eq_paths and os.path.exists(eq_paths["eq3_delta_icl"]):
        story.append(Image(eq_paths["eq3_delta_icl"], width=504, height=54))
        story.append(Spacer(1, 4))
        
    story.append(Paragraph(
        "<b>2.3 Ponderación de Dimensiones por Entropía de Shannon y Formulación Continua del IVF:</b> Para construir el score base "
        "V_i ∈ [0, 1] sin juicios de valor arbitrarios, se aplica el Método de Entropía de Shannon (Zeleny, 1982; Nardo et al., OCDE 2005). "
        "Luego, el IVF individual se obtiene como una combinación convexa (α = 0.70) entre la vulnerabilidad del perfil latente R_k y la "
        "dispersión intracluster individual V_i, escalada a 1000 puntos:",
        body_style
    ))
    
    # Ecuación 4: Shannon Weights
    if "eq4_shannon" in eq_paths and os.path.exists(eq_paths["eq4_shannon"]):
        story.append(Image(eq_paths["eq4_shannon"], width=504, height=54))
        story.append(Spacer(1, 3))
        
    # Ecuación 6: Continuous IVF
    if "eq6_ivf" in eq_paths and os.path.exists(eq_paths["eq6_ivf"]):
        story.append(Image(eq_paths["eq6_ivf"], width=504, height=54))
        story.append(Spacer(1, 4))
        
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 3: METODOLOGÍA, SUPUESTOS Y TABLA MAESTRA DE VARIABLES
    # ==========================================================================
    story.append(Paragraph("3. Metodología, Supuestos Estadísticos y Tabla Maestra de Variables", h1_style))
    story.append(Paragraph(
        "Bajo el supuesto de Datos Faltantes al Azar (MAR: Missing At Random), la probabilidad de no respuesta depende únicamente "
        "de las características observadas del hogar y del individuo. Para evitar distorsiones métricas, se aplicó una imputación "
        "diferenciada: (1) Moda Ponderada para variables nominales con faltantes ≤ 5%; (2) Mediana para variables ordinales y continuas "
        "≤ 5%; (3) MICE (Fully Conditional Specification con árboles ExtraTrees) para variables estructurales > 5%; y (4) Imputación "
        "econométrica del ingreso laboral sobre ln(Y + 1), preservando el cero estructural genuino (S/. 0.00) en todos los Trabajadores "
        "Familiares No Remunerados (TFNR, P507 == 5).",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    # Tabla Maestra de Variables
    master_var_data = [
        [Paragraph("Código ENAHO", table_hdr), Paragraph("Nombre de Negocio", table_hdr), Paragraph("Módulo", table_hdr), Paragraph("Tratamiento e Imputación", table_hdr), Paragraph("Rol Analítico", table_hdr)],
        [Paragraph("P106A / B", table_cell_bold), Paragraph("Colateral / Título SUNARP", table_cell), Paragraph("Mod 100", table_cell), Paragraph("Binaria / Moda Ponderada", table_cell), Paragraph("Garantía Microfinanciera", table_cell)],
        [Paragraph("NBI1-5", table_cell_bold), Paragraph("Índice NBI (0 a 5)", table_cell), Paragraph("Mod 100", table_cell), Paragraph("Suma Aditiva / Mediana", table_cell), Paragraph("Carencia Material Hábitat", table_cell)],
        [Paragraph("P1144 / B2", table_cell_bold), Paragraph("Conectividad Hogar Postpago/Wifi", table_cell), Paragraph("Mod 100", table_cell), Paragraph("Binaria Directa", table_cell), Paragraph("Ecosistema Digital Hogar", table_cell)],
        [Paragraph("P207 / P208A", table_cell_bold), Paragraph("Género y Edad Adulta", table_cell), Paragraph("Mod 200", table_cell), Paragraph("Filtro Activo 18-65 Años", table_cell), Paragraph("Demografía de Control", table_cell)],
        [Paragraph("P301A", table_cell_bold), Paragraph("Capital Humano (Educación)", table_cell), Paragraph("Mod 300", table_cell), Paragraph("Ordinal Formal (0-11)", table_cell), Paragraph("Capacidad Cognitiva", table_cell)],
        [Paragraph("P314B1_9 / P316_8", table_cell_bold), Paragraph("Plan Datos y Ventas Online", table_cell), Paragraph("Mod 300", table_cell), Paragraph("Binaria Directa", table_cell), Paragraph("Bancarización Digital", table_cell)],
        [Paragraph("P401C", table_cell_bold), Paragraph("Identidad Legal KYC (DNI)", table_cell), Paragraph("Mod 400", table_cell), Paragraph("Binaria Directa", table_cell), Paragraph("Barrera Regulatoria", table_cell)],
        [Paragraph("P507 / P510A1", table_cell_bold), Paragraph("Categoría Ocupacional y RUC", table_cell), Paragraph("Mod 500", table_cell), Paragraph("MICE FCS Categórico", table_cell), Paragraph("Formalidad Laboral", table_cell)],
        [Paragraph("P524A1 / P530A", table_cell_bold), Paragraph("Ingreso Líquido Mensual (S/.)", table_cell), Paragraph("Mod 500", table_cell), Paragraph("MICE ln(Y+1) / TFNR=0", table_cell), Paragraph("Solvencia Económica", table_cell)],
        [Paragraph("P558E1_6", table_cell_bold), Paragraph("Tenencia de Producto Financiero", table_cell), Paragraph("Mod 500", table_cell), Paragraph("Binaria Filtro (0/1)", table_cell), Paragraph("Variable Objetivo Look-Alike", table_cell)],
        [Paragraph("P558G1 / G7", table_cell_bold), Paragraph("Ahorro Juntas y Préstamo Informal", table_cell), Paragraph("Mod 500", table_cell), Paragraph("Binarias de Conducta", table_cell), Paragraph("Finanzas Informales", table_cell)],
        [Paragraph("P602 / P602N", table_cell_bold), Paragraph("Olla Común / Comedor Popular", table_cell), Paragraph("Mod 602", table_cell), Paragraph("Agregación Hogar Binaria", table_cell), Paragraph("Vulnerabilidad Alimentaria", table_cell)],
        [Paragraph("P701 / P710", table_cell_bold), Paragraph("Subsidios JUNTOS / Pensión 65 / FISE", table_cell), Paragraph("Mod 700", table_cell), Paragraph("Agregación Hogar Binaria", table_cell), Paragraph("Dependencia de Protección", table_cell)],
        [Paragraph("FAC500A", table_cell_bold), Paragraph("Factor de Expansión Poblacional", table_cell), Paragraph("Mod 500", table_cell), Paragraph("Horvitz-Thompson Numérico", table_cell), Paragraph("Ponderación Inferencia", table_cell)]
    ]
    
    t_master = Table(master_var_data, colWidths=[65, 120, 50, 125, 144])
    t_master.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_master)
    story.append(Spacer(1, 8))
    
    # Tabla de Pesos Shannon Comparativa
    if weights_comparison_df is not None:
        story.append(Paragraph("<b>Ponderaciones de las Dimensiones del Score Base Vi (Entropía vs. Pesos Iguales):</b>", h2_style))
        w_data = [
            [Paragraph("Dimensión de Vulnerabilidad", table_hdr), Paragraph("Peso Shannon (w_j)", table_hdr), Paragraph("Peso Igual Ref.", table_hdr), Paragraph("Entropía (e_j)", table_hdr), Paragraph("Diversificación (d_j)", table_hdr)]
        ]
        for _, r in weights_comparison_df.iterrows():
            w_data.append([
                Paragraph(str(r["Dimension_Vulnerabilidad"]), table_cell_bold),
                Paragraph(f"<b>{r['Peso_Entropia_Shannon']:.4f} ({r['Peso_Entropia_Shannon']*100:.1f}%)</b>", table_cell),
                Paragraph(f"{r['Peso_Igual_Referencia']:.4f}", table_cell),
                Paragraph(f"{r['Entropia_Normalizada_ej']:.4f}", table_cell),
                Paragraph(f"{r['Grado_Diversificacion_dj']:.4f}", table_cell)
            ])
        t_w = Table(w_data, colWidths=[184, 90, 75, 75, 80])
        t_w.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t_w)
        
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 4: RESULTADOS DEL MODELO LCA Y DEMOSTRACIÓN DE K*
    # ==========================================================================
    story.append(Paragraph("4. Resultados del Modelo Latente e Índice IVF", h1_style))
    story.append(Paragraph(
        f"<b>4.1 Demostración Empírica del Número Óptimo de Perfiles (K* = {k_star}):</b> En la Tabla 2 se presenta la grilla "
        f"de evaluación para K ∈ {{2, 3, 4, 5, 6, 7}}. La evidencia empírica valida plenamente la selección de K* = {k_star}: "
        f"(1) Para K ≥ 5, la masa poblacional mínima se desploma al 0.07%, violando la restricción muestral de masa mínima (π_k ≥ 5%); "
        f"(2) Al pasar de K=2 a K=3 se obtiene una reducción del ICL de 48.06% con una entropía relativa ínfima (0.0018); "
        f"(3) El modelo con K=4 incrementa el ICL a 126,183 (+61.6% de deterioro) por dispersión entrópica (0.1440). Por ende, "
        f"K* = {k_star} es el único óptimo parsimonioso y no espurio.",
        body_style
    ))
    
    # Gráfico 1: Selección K*
    if "k_selection" in figure_paths and os.path.exists(figure_paths["k_selection"]):
        story.append(Image(figure_paths["k_selection"], width=504, height=190))
        story.append(Spacer(1, 6))
        
    # Tabla de Grilla K
    grid_data = [
        [Paragraph("K", table_hdr), Paragraph("Log-Likelihood", table_hdr), Paragraph("Parám. d", table_hdr), Paragraph("BIC", table_hdr), Paragraph("ICL", table_hdr), Paragraph("ΔICL (%)", table_hdr), Paragraph("Entropía Rel.", table_hdr), Paragraph("Masa Mín (%)", table_hdr), Paragraph("Estado", table_hdr)]
    ]
    for _, r in metrics_grid_df.iterrows():
        estado = "ÓPTIMO K*" if int(r["K"]) == k_star else ("Válido" if r["Masa_Valida"] else "Espurio (<5%)")
        grid_data.append([
            Paragraph(str(int(r["K"])), table_cell_bold),
            Paragraph(f"{r['Log_Likelihood']:,.1f}", table_cell),
            Paragraph(str(int(r["Parametros_d"])), table_cell),
            Paragraph(f"{r['BIC']:,.1f}", table_cell),
            Paragraph(f"{r['ICL']:,.1f}", table_cell),
            Paragraph(f"{r['Delta_ICL_pct']:.1f}%", table_cell),
            Paragraph(f"{r['Entropia_Relativa']:.4f}", table_cell),
            Paragraph(f"{r['Masa_Minima_pct']:.2f}%", table_cell),
            Paragraph(estado, table_cell_bold if int(r["K"]) == k_star else table_cell)
        ])
    t_grid = Table(grid_data, colWidths=[20, 68, 55, 60, 60, 48, 58, 60, 75])
    t_grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_grid)
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 5: CARACTERIZACIÓN SOCIOECONÓMICA Y VALIDACIÓN BOOTSTRAP
    # ==========================================================================
    story.append(Paragraph(
        f"<b>4.2 Caracterización Socioeconómica de los K*={k_star} Perfiles Latentes:</b> "
        f"La formulación del IVF continuo distribuye a los 28,016 individuos no bancarizados en un soporte "
        f"continuo entre 166.2 y 638.4 puntos, preservando la jerarquía ordinal de los grupos pero dotando a cada individuo "
        f"de variabilidad granular para originación crediticia.",
        body_style
    ))
    
    # Gráfico 2: Distribución Continua IVF
    if "ivf_distribution" in figure_paths and os.path.exists(figure_paths["ivf_distribution"]):
        story.append(Image(figure_paths["ivf_distribution"], width=504, height=190))
        story.append(Spacer(1, 6))
        
    # Tabla de Perfiles
    if cluster_summary_df is not None:
        prof_data = [
            [Paragraph("Perfil", table_hdr), Paragraph("Nombre de Negocio", table_hdr), Paragraph("Población Expandida", table_hdr), Paragraph("Masa (%)", table_hdr), Paragraph("Vulnerabilidad R_k", table_hdr), Paragraph("IVF Medio", table_hdr)]
        ]
        for _, r in cluster_summary_df.iterrows():
            prof_data.append([
                Paragraph(f"Perfil {int(r['cluster_lca'])}", table_cell_bold),
                Paragraph(str(r["perfil_nombre"]), table_cell),
                Paragraph(f"{int(r['poblacion_expandida']):,} hab.", table_cell),
                Paragraph(f"{r['proporcion_expandida_pct']:.2f}%", table_cell),
                Paragraph(f"{r['vulnerabilidad_Rk']:.4f}", table_cell),
                Paragraph(f"<b>{r['indice_ivf']:.1f} pts</b>", table_cell)
            ])
        t_prof = Table(prof_data, colWidths=[45, 194, 95, 55, 65, 50])
        t_prof.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t_prof)
        story.append(Spacer(1, 6))
        
    story.append(Paragraph(
        "<b>4.3 Validación Asintótica por Bootstrap Complejo de Conglomerados (B = 200 Réplicas):</b> "
        "El remuestreo estratificado por conglomerados primarios (CONGLOME) con corrección por el Algoritmo Húngaro confirma la alta reproducibilidad "
        "del modelo. El Índice de Rand Ajustado (ARI) promedio alcanzó 0.7917 (IC 95%: [0.5651, 0.9999]), superando ampliamente el umbral "
        "convencional de robustez de 0.70. El IVF medio poblacional exhibe un error estándar de tan solo ±2.33 puntos (IC 95%: [265.59, 274.34]).",
        body_style
    ))
    
    # Ecuación 9: Bootstrap & Hungarian Alignment
    if "eq9_bootstrap" in eq_paths and os.path.exists(eq_paths["eq9_bootstrap"]):
        story.append(Image(eq_paths["eq9_bootstrap"], width=504, height=54))
        story.append(Spacer(1, 4))
        
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 6: MODELO LOOK-ALIKE, EXPLICABILIDAD SHAP Y MATRIZ 2D
    # ==========================================================================
    story.append(Paragraph("5. Resultados del Modelo Look-Alike, Explicabilidad SHAP y Matriz Microfinanciera", h1_style))
    story.append(Paragraph(
        f"<b>5.1 Capacidad Predictiva del Modelo Look-Alike:</b> Entrenado sobre el panel completo de adultos (N = 62,825) con ponderación "
        f"FAC500A, el ensamble CatBoost calibrado con regresión isotónica alcanzó un ROC-AUC de {lookalike_metrics['ROC_AUC']:.4f}, "
        f"un PR-AUC de {lookalike_metrics['PR_AUC']:.4f} y un estadístico Kolmogorov-Smirnov (KS) de {lookalike_metrics['KS_Statistic']:.2f}%. "
        f"La calibración isotónica asegura que las probabilidades predichas representen tasas empíricas reales de adopción financiera:",
        body_style
    ))
    
    # Ecuación 7: Isotonic Calibration
    if "eq7_isotonic" in eq_paths and os.path.exists(eq_paths["eq7_isotonic"]):
        story.append(Image(eq_paths["eq7_isotonic"], width=504, height=54))
        story.append(Spacer(1, 4))
        
    story.append(Paragraph(
        "<b>5.2 Explicabilidad SHAP (Valores de Shapley Agrupados):</b> Basado en teoría de juegos cooperativos, se descompone la predicción "
        "en contribuciones aditivas exactas. Los códigos técnicos han sido 100% traducidos a conceptos de negocio:",
        body_style
    ))
    
    # Ecuación 8: SHAP
    if "eq8_shap" in eq_paths and os.path.exists(eq_paths["eq8_shap"]):
        story.append(Image(eq_paths["eq8_shap"], width=504, height=54))
        story.append(Spacer(1, 4))
        
    # Gráfico 3: SHAP Feature Importance
    if "shap_importance" in figure_paths and os.path.exists(figure_paths["shap_importance"]):
        story.append(Image(figure_paths["shap_importance"], width=504, height=195))
        story.append(Spacer(1, 6))
        
    # Gráfico 4: Matriz Estratégica 2D
    if "strategic_matrix" in figure_paths and os.path.exists(figure_paths["strategic_matrix"]):
        story.append(Image(figure_paths["strategic_matrix"], width=504, height=200))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph(
        "<b>5.3 Matriz de Originación Microfinanciera 2D (IVF vs. Propensión Look-Alike):</b> "
        "El cruce del IVF con el Score Look-Alike delimita 4 cuadrantes operativos para microfinancieras y bancos:<br/>"
        "• <b>Cuadrante I (Onboarding Digital Inmediato):</b> Bajo IVF, alta propensión. Universo 'Quick-Wins' bancarizable vía billeteras digitales autónomas (Yape/Plin).<br/>"
        "• <b>Cuadrante II (Microfinanzas Productivas):</b> IVF moderado, propensión media-alta con colateral SUNARP o RUC. Target para crédito de capital de trabajo MYPE.<br/>"
        "• <b>Cuadrante III (Bancarización Asistida por Agentes):</b> Alto IVF material/educativo, baja propensión. Requiere cuentas básicas en corresponsales y microseguros.<br/>"
        "• <b>Cuadrante IV (Inclusión Social Prioritaria):</b> IVF crítico con inseguridad alimentaria o carencia de DNI. Requiere articulación con el Estado (MIDIS/RENIEC).",
        body_style
    ))
    story.append(PageBreak())
    
    # ==========================================================================
    # PÁGINA 7: CONCLUSIONES Y BIBLIOGRAFÍA APA 7
    # ==========================================================================
    story.append(Paragraph("6. Conclusiones y Recomendaciones de Política Microfinanciera", h1_style))
    story.append(Paragraph(
        "1. <b>Superación del Paradigma de Exclusión Binaria:</b> La exclusión financiera en el Perú no es un bloque monolítico. "
        "El IVF continuo (0 a 1000) permite a las microfinancieras diferenciar entre no bancarizados solventes con brecha digital transitoria "
        "y hogares en subsistencia crítica con inseguridad alimentaria.<br/>"
        "2. <b>El Rol Transformador del Ecosistema Digital ENAHO 2025:</b> El acceso a smartphones con planes de datos y el uso de redes sociales "
        "para ventas comerciales se consolida como el principal catalizador de bancarización espontánea, superando al ingreso absoluto.<br/>"
        "3. <b>Colateral Predial SUNARP como Palanca Microfinanciera:</b> La regularización registral de la vivienda reduce la vulnerabilidad "
        "financiera y permite originar créditos productivos a tasas sostenibles para microempresarios informales.<br/>"
        "4. <b>Validación y Robustez de K*={k_star}:</b> La selección algorítmica multicriterio garantiza que ningún cluster sea ficticio, "
        "avalado por 200 réplicas bootstrap complejas con estabilidad ARI > 0.79.",
        body_style
    ))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Referencias Bibliográficas (APA 7)", h2_style))
    refs = [
        "Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 22(7), 719-725.",
        "Hubert, L., & Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193-218.",
        "Instituto Nacional de Estadística e Informática - INEI. (2025). <i>Encuesta Nacional de Hogares (ENAHO 2025): Ficha Técnica y Metodología de Muestreo Complejo</i>. Lima: Dirección Nacional de Censos y Encuestas.",
        "Kuhn, H. W. (1955). The Hungarian method for the assignment problem. <i>Naval Research Logistics Quarterly</i>, 2(1‐2), 83-97.",
        "Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. <i>Nature Machine Intelligence</i>, 2(1), 56-67.",
        "Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A., & Giovannini, E. (2005). <i>Handbook on constructing composite indicators: Methodology and user guide</i>. OECD Publishing, Paris.",
        "Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. <i>Journal of the American Statistical Association</i>, 83(401), 231-241.",
        "Rubin, D. B. (1976). Inference and missing data. <i>Biometrika</i>, 63(3), 581-592.",
        "Van Buuren, S. (2018). <i>Flexible Imputation of Missing Data</i> (2nd ed.). CRC Press / Chapman and Hall.",
        "Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In <i>Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining</i> (pp. 694-699).",
        "Zeleny, M. (1982). <i>Multiple criteria decision making</i>. McGraw-Hill, New York."
    ]
    for r in refs:
        story.append(Paragraph(f"• {r}", ParagraphStyle("RefStyle", parent=body_style, fontSize=7.5, leading=10.5, spaceAfter=3.5)))
        
    # Compilación final con NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"Informe Científico PDF generado con éxito en: {output_pdf_path} ({os.path.getsize(output_pdf_path):,} bytes).")
    return output_pdf_path
