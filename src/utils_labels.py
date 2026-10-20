"""
================================================================================
MÓDULO: src/utils_labels.py
DICCIONARIO MAESTRO DE TRADUCCIÓN A LENGUAJE DE NEGOCIO Y GENERADOR DE NOMBRES
================================================================================
Este módulo garantiza la regla transversal de 'Cero Códigos Crudos de ENAHO'
en todos los entregables (Dashboard HTML, Informe PDF, gráficos SHAP y tablas).
Además, implementa el asignador autónomo de nombres de negocio a los perfiles
latentes K* en función de sus características socioeconómicas empíricas.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Diccionario maestro de traducción de variables crudas y derivadas
VARIABLE_BUSINESS_LABELS: Dict[str, str] = {
    # Módulo 100 - Vivienda, Garantías y Conectividad
    "P106A": "Tenencia de Título de Propiedad",
    "P106B": "Título Registrado en SUNARP",
    "tiene_colateral_propiedad": "Colateral / Garantía Predial SUNARP",
    "NBI1": "Carencia de Vivienda Adecuada (NBI 1)",
    "NBI2": "Hacinamiento Crítico (NBI 2)",
    "NBI3": "Carencia de Servicios Higiénicos (NBI 3)",
    "NBI4": "Inasistencia Escolar Infantil (NBI 4)",
    "NBI5": "Alta Dependencia Económica (NBI 5)",
    "nbi_score": "Índice de Necesidades Básicas Insatisfechas (0-5)",
    "P1141": "Teléfono Fijo en el Hogar",
    "P1142": "Teléfono Celular en el Hogar",
    "P1144": "Acceso a Internet Fijo/Wifi en Hogar",
    "P114B1": "Telefonía Fija Residencial",
    "P114B2": "Plan Móvil Postpago en Hogar",
    "P114B3": "Móvil Prepago en Hogar",
    "DOMINIO": "Dominio Geográfico ENAHO",
    "ESTRATO": "Estrato Demográfico / Tamaño Poblacional",
    "area": "Área de Residencia (Urbano / Rural)",
    "dominio_macro": "Macrorregión Geográfica",
    "es_urbano": "Residencia en Zona Urbana",

    # Módulo 200 - Demografía y Composición del Hogar
    "P207": "Género del Individuo",
    "es_mujer": "Sexo Femenino",
    "P208A": "Edad en Años Cumplidos",
    "edad": "Edad del Titular (Años)",
    "P209": "Estado Civil",
    "P203": "Parentesco con Jefe de Hogar",
    "es_jefe_hogar": "Jefatura de Hogar",

    # Módulo 300 - Capital Humano y Ecosistema Digital
    "P300A": "Lengua Materna",
    "es_lengua_originaria": "Lengua Materna Indígena / Originaria",
    "P301A": "Nivel Educativo Alcanzado",
    "educacion_anios": "Nivel Educativo (Ordinal Formal)",
    "nivel_educativo_desc": "Máximo Nivel de Instrucción",
    "P314A": "Uso Personal de Internet",
    "usa_internet": "Uso Activo de Internet",
    "P314B1_9": "Conectividad Celular con Plan de Datos",
    "tiene_plan_datos": "Conectividad Móvil con Datos",
    "P316A1": "Tenencia Individual de Smartphone Propio",
    "tiene_celular_propio": "Tenencia de Celular Propio",
    "P316$8": "Uso de Internet para Ventas Online / Redes",
    "P316_8": "Comercialización Digital y Redes Sociales",
    "ventas_digitales": "Comercialización Digital y Redes Sociales",
    "P316B": "Uso de Computadora / Laptop",
    "usa_computadora": "Uso Frecuente de Computadora",
    "acceso_digital": "Índice de Inclusión Digital (0 a 5)",

    # Módulo 400 - Identidad Legal KYC y Protección de Salud
    "P401C": "Posesión de Documento Nacional de Identidad (DNI)",
    "tiene_dni": "Identidad Legal KYC (Tenencia de DNI)",
    "sin_dni_barrera": "Barrera Legal KYC (Carece de DNI)",
    "P4191": "Afiliación a Seguro de Salud EsSalud",
    "seguro_essalud": "Afiliación a EsSalud (Seguro Contributivo)",
    "P4195": "Afiliación al Seguro Integral de Salud (SIS)",
    "seguro_sis": "Afiliación al SIS (Seguro Subsidiado)",

    # Módulo 500 - Estructura Laboral, Ingresos e Inclusión Financiera
    "P507": "Categoría Ocupacional ENAHO",
    "categoria_ocupacional_desc": "Inserción Ocupacional",
    "P510A1": "Registro Tributario SUNAT (RUC de Negocio)",
    "tiene_ruc": "Negocio Formalizado con RUC (SUNAT)",
    "P511A": "Modalidad de Contrato Laboral",
    "P512A": "Tamaño de Empresa / Empleador (N° Trabajadores)",
    "informalidad": "Condición de Informalidad Laboral",
    "empleo_vulnerable": "Inserción Laboral Vulnerable (Indep. No Calificado/TFNR)",
    "ingreso_mensual": "Ingreso Total Mensual (S/.)",
    "log_ingreso": "Logaritmo del Ingreso Mensual ln(Y + 1)",
    "P524A1": "Ingreso Líquido Trabajo Asalariado",
    "P530A": "Ganancia Líquida Negocio Independiente",
    "P558A5": "Sin Afiliación a Pensiones (AFP / ONP)",
    "sin_pension": "Desprotección Previsional (Sin AFP/ONP)",
    "P558C": "Autoidentificación Étnica",
    "P558E1_1": "Posee Cuenta de Ahorros Formal",
    "P558E1_6": "Condición de No Bancarizado ENAHO",
    "IS_BANCARIZADO": "Titular de Producto Financiero Formal",
    "P558F1A": "Barrera Principal de Acceso a Cuenta Bancaria",
    "P558F1B": "Barrera Principal de Acceso a Crédito",
    "P558G1": "Ahorro Informal en Juntas o Panderos",
    "ahorro_juntas": "Práctica de Ahorro en Juntas Comunitarias",
    "P558G2": "Ahorro Custodiado por Familiares / Conocidos",
    "ahorro_familiares": "Ahorro Informal con Familiares",
    "P558G3": "Ahorro en Efectivo en Casa (Bajo el Colchón)",
    "ahorro_casa": "Preferencia por Efectivo en Hogar",
    "P558G7": "Financiamiento de Prestamista Informal (Gota a Gota)",
    "prestamo_informal": "Exposición a Crédito Informal / Usura",
    "FAC500A": "Factor de Expansión Poblacional (Horvitz-Thompson)",

    # Módulos 602 y 700 - Inseguridad Alimentaria y Protección Social
    "P602N": "Entidad Benéfica Alimentaria",
    "P602": "Recepción de Apoyo Alimentario",
    "acude_olla_comedor": "Dependencia Alimentaria (Ollas Comunes / Comedores)",
    "P710$04": "Beneficiario Programa JUNTOS",
    "recibe_juntos": "Transferencia Condicionada Programa JUNTOS",
    "P710$05": "Beneficiario Programa Pensión 65",
    "recibe_pension65": "Subsidio No Contributivo Pensión 65",
    "P701$03": "Beneficiario Programa Wasi Mikuna",
    "recibe_wasi_mikuna": "Apoyo Nutricional Wasi Mikuna",
    "P710$15": "Beneficiario Bono Gas FISE",
    "recibe_bono_fise": "Subsidio Energético Bono Gas FISE",
    "recibe_subsidio_estatal": "Recepción de Subsidios o Transferencias Estatales",

    # Métricas Analíticas del Modelo Latente y Look-Alike
    "cluster_lca": "Perfil Latente de Vulnerabilidad Asignado",
    "perfil_nombre": "Perfil Socioeconómico Latente",
    "score_vulnerabilidad_base": "Score Base de Carencia Vi (0-1)",
    "indice_ivf": "Índice de Vulnerabilidad Financiera (IVF 0-1000)",
    "ivf_estandarizado": "Índice IVF Estandarizado",
    "propensity_score": "Score de Propensión a Bancarización (0% - 100%)",
    "cuadrante_estrategico": "Cuadrante de Inclusión Microfinanciera",
    "estrategia_recomendada": "Estrategia Microfinanciera Prescriptiva",
    "producto_recomendado": "Solución Financiera Personalizada",
    "canal_prioritario": "Canal de Atención Óptimo"
}


def get_business_label(variable_name: str) -> str:
    """
    Retorna la etiqueta de negocio para cualquier variable, función o columna.
    Garantiza que nunca se filtre un código crudo de ENAHO en reportes.
    """
    clean_name = variable_name.replace("$", "_").strip()
    if clean_name in VARIABLE_BUSINESS_LABELS:
        return VARIABLE_BUSINESS_LABELS[clean_name]
    if variable_name in VARIABLE_BUSINESS_LABELS:
        return VARIABLE_BUSINESS_LABELS[variable_name]
    
    # Tratamiento de dummies de SHAP: e.g. "dominio_macro_Costa" -> "Macrorregión: Costa"
    for prefix in ["dominio_macro_", "area_", "nivel_educativo_desc_", "categoria_ocupacional_desc_"]:
        if variable_name.startswith(prefix):
            val = variable_name[len(prefix):]
            base = VARIABLE_BUSINESS_LABELS.get(prefix[:-1], prefix[:-1])
            return f"{base}: {val}"
            
    # Formato legible si no se encuentra
    return clean_name.replace("_", " ").title()


def generate_profile_names(summary_df: pd.DataFrame) -> Dict[int, str]:
    """
    Asigna nombres de negocio descriptivos, interpretativos y no redundantes
    a cada cluster k = 0, ..., K-1 en función de sus características empíricas ponderadas.
    
    Analiza:
    - Ingreso mensual promedio
    - Nivel de digitalización (acceso_digital)
    - Formalidad laboral (RUC, informalidad)
    - Carencias materiales (NBI)
    - Vulnerabilidad alimentaria y dependencia de subsidios estatales
    - Colateral / propiedad inmobiliaria
    """
    k_profiles = summary_df.index.tolist()
    names: Dict[int, str] = {}
    assigned_names = set()
    
    # Calcular rangos y percentiles relativos entre clusters
    ingreso_rank = summary_df["ingreso_mensual"].rank(ascending=False)
    digital_rank = summary_df["acceso_digital"].rank(ascending=False)
    nbi_rank = summary_df["nbi_score"].rank(ascending=False)
    alimento_rank = summary_df["acude_olla_comedor"].rank(ascending=False) if "acude_olla_comedor" in summary_df.columns else pd.Series(0, index=summary_df.index)
    colateral_rank = summary_df["tiene_colateral_propiedad"].rank(ascending=False) if "tiene_colateral_propiedad" in summary_df.columns else pd.Series(0, index=summary_df.index)
    subsidio_rank = summary_df["recibe_subsidio_estatal"].rank(ascending=False) if "recibe_subsidio_estatal" in summary_df.columns else pd.Series(0, index=summary_df.index)

    for k in k_profiles:
        ing_val = summary_df.loc[k, "ingreso_mensual"]
        dig_val = summary_df.loc[k, "acceso_digital"]
        nbi_val = summary_df.loc[k, "nbi_score"]
        alim_val = summary_df.loc[k, "acude_olla_comedor"] if "acude_olla_comedor" in summary_df.columns else 0
        colat_val = summary_df.loc[k, "tiene_colateral_propiedad"] if "tiene_colateral_propiedad" in summary_df.columns else 0
        sub_val = summary_df.loc[k, "recibe_subsidio_estatal"] if "recibe_subsidio_estatal" in summary_df.columns else 0

        # Reglas multicriterio jerárquicas
        if alimento_rank[k] == 1 and alim_val > 0.08:
            candidate = "Vulnerabilidad Crítica e Inseguridad Alimentaria"
        elif nbi_rank[k] == 1 and nbi_val >= 1.0:
            candidate = "Subsistencia Rural con Alta Privación Material"
        elif ingreso_rank[k] == 1 and digital_rank[k] <= 2:
            candidate = "Microempresarios Digitalizados con Alto Potencial"
        elif colateral_rank[k] <= 2 and colat_val >= 0.40:
            candidate = "Emprendedores Tradicionales con Colateral Predial"
        elif digital_rank[k] <= 2 and ing_val >= 1200:
            candidate = "Jóvenes Conectados en Transición Laboral"
        elif sub_val > 0.30 or subsidio_rank[k] == 1:
            candidate = "Hogares Receptores de Protección Social Estatal"
        elif ing_val < 900:
            candidate = "Trabajadores Informales de Baja Productividad"
        else:
            candidate = f"Microemprendimiento Informal de Ingreso Medio"

        # Resolver colisiones si dos clusters coinciden
        counter = 2
        final_name = candidate
        while final_name in assigned_names:
            final_name = f"{candidate} (Segmento {counter})"
            counter += 1
            
        assigned_names.add(final_name)
        names[k] = final_name

    return names
