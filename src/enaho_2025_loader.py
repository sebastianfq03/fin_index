"""
================================================================================
MÓDULO: src/enaho_2025_loader.py
ETAPA 1: INGENIERÍA DE DATOS E INCORPORACIONES ENAHO 2025 (OPTIMIZADO)
================================================================================
Fundamentación Teórica y Metodológica de Encuestas Complejas:
--------------------------------------------------------------------------------
La Encuesta Nacional de Hogares (ENAHO 2025), conducida por el INEI, utiliza
un diseño muestral probabilístico, bietápico, estratificado e independiente en
cada departamento del Perú. Para la inferencia poblacional de la vulnerabilidad
financiera y la propensión a la bancarización, este módulo realiza:

1. Integración Relacional Jerárquica:
   - Nivel Hogar: Identificado unívocamente por [CONGLOME, VIVIENDA, HOGAR].
     Consolida las características estructurales del hábitat (Módulo 100),
     la inseguridad alimentaria crítica (Módulo 602: Ollas Comunes y Comedores)
     y la red de protección social y subsidios del Estado (Módulo 700: JUNTOS,
     Pensión 65, Wasi Mikuna y Bono Gas FISE).
   - Nivel Individuo: Identificado por [CONGLOME, VIVIENDA, HOGAR, CODPERSO].
     Integra la demografía (Módulo 200), capital humano y ecosistema digital
     (Módulo 300), identidad legal KYC y aseguramiento en salud (Módulo 400),
     y la inserción laboral, ingresos e inclusión financiera (Módulo 500).

2. Delimitación de Universos Analíticos:
   - Panel Total Adulto (18 a 65 años en P208A): Universo muestral con respuesta
     financiera válida en el Módulo 500. Se define la variable binaria
     IS_BANCARIZADO (1 = Posee producto formal de ahorro, cuenta o tarjeta;
     0 = No posee producto formal, P558E1_6 == 6). Este panel entrena el Modelo 2
     (Look-Alike Propensity Model).
   - Población Objetivo No Bancarizada (P558E1_6 == 6 & 18 <= P208A <= 65):
     Subconjunto de exclusión financiera formal sobre el cual se estima el Modelo 1
     (LCA e Índice de Vulnerabilidad Financiera IVF) y sobre el cual se realiza la
     predicción de propensión a bancarización.

3. Factor de Expansión Poblacional (FAC500A):
   En encuestas complejas con probabilidades desiguales de selección, la estimación
   de totales y promedios poblacionales requiere el estimador de Horvitz-Thompson:
       \\hat{\\theta}_{HT} = \\frac{\\sum_{i=1}^n w_i y_i}{\\sum_{i=1}^n w_i}, \\quad w_i = FAC500A_i
   Este factor es limpiado rigurosamente (conversión de coma decimal) y propagado
   en todos los análisis posteriores.
"""

import os
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_keys(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Limpia y castea las llaves de integración a formatos estandarizados."""
    for col in key_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    return df


def load_and_aggregate_food_aid(file_path: str) -> pd.DataFrame:
    """
    Agrega el Módulo 602 (Instituciones Benéficas y Alimentarias) a nivel Hogar.
    ENAHO 2025 captura las Ollas Comunes (P602N in [14, 15, 16]), Comedores
    Populares y Vaso de Leche (P602N in [1, 3] o cualquier P602 == 1).
    Retorna indicador binario acude_olla_comedor a nivel ID_HOGAR.
    """
    logger.info(f"Procesando Módulo 602 de vulnerabilidad alimentaria: {file_path}")
    cols_to_use = ["CONGLOME", "VIVIENDA", "HOGAR", "P602N", "P602"]
    
    df602 = pd.read_csv(
        file_path,
        sep=";",
        usecols=cols_to_use,
        encoding="latin1",
        low_memory=False
    )
    df602 = _normalize_keys(df602, ["CONGLOME", "VIVIENDA", "HOGAR"])
    df602["P602"] = pd.to_numeric(df602["P602"], errors="coerce").fillna(0)
    df602["P602N"] = pd.to_numeric(df602["P602N"], errors="coerce").fillna(0)
    
    is_food_aid = (df602["P602"] == 1) & (
        df602["P602N"].isin([1, 2, 3, 14, 15, 16]) | (df602["P602"] == 1)
    )
    
    food_aid_households = (
        df602[is_food_aid]
        .groupby(["CONGLOME", "VIVIENDA", "HOGAR"])
        .size()
        .reset_index(name="count_ayuda_alim")
    )
    food_aid_households["acude_olla_comedor"] = 1
    return food_aid_households[["CONGLOME", "VIVIENDA", "HOGAR", "acude_olla_comedor"]]


def load_and_aggregate_subsidies(file_path: str) -> pd.DataFrame:
    """
    Agrega el Módulo 700 (Programas Sociales y Transferencias) a nivel Hogar.
    Identifica beneficiarios de:
    - JUNTOS (P710$04 == 1)
    - Pensión 65 (P710$05 == 1)
    - Wasi Mikuna (P701$03 == 1 o P701$04 == 1)
    - Bono Gas FISE (P710$15 == 1)
    Genera indicadores específicos e indicador agregado recibe_subsidio_estatal.
    """
    logger.info(f"Procesando Módulo 700 de transferencias y subsidios: {file_path}")
    
    # Inspeccionar columnas para cargar sólo las necesarias
    header = pd.read_csv(file_path, sep=";", nrows=0, encoding="latin1").columns.tolist()
    clean_map = {c: c.replace("$", "_").strip() for c in header}
    
    # Determinar qué columnas leer
    id_cols = ["CONGLOME", "VIVIENDA", "HOGAR"]
    target_raw_cols = [c for c in header if c in id_cols or clean_map[c] in [
        "P710_04", "P710_05", "P710_15", "P701_03", "P701_04"
    ]]
    
    df700 = pd.read_csv(file_path, sep=";", usecols=target_raw_cols, encoding="latin1", low_memory=False)
    df700.columns = [clean_map[c] for c in df700.columns]
    df700 = _normalize_keys(df700, id_cols)
    
    # Identificar columnas
    col_juntos = [c for c in df700.columns if "P710" in c and "04" in c]
    col_pension65 = [c for c in df700.columns if "P710" in c and "05" in c]
    col_fise = [c for c in df700.columns if "P710" in c and "15" in c]
    col_wasi = [c for c in df700.columns if "P701" in c and ("03" in c or "04" in c)]
    
    for c in col_juntos + col_pension65 + col_fise + col_wasi:
        df700[c] = pd.to_numeric(df700[c], errors="coerce").fillna(0).astype(int)
        
    df700["recibe_juntos"] = (df700[col_juntos].sum(axis=1) > 0).astype(int) if col_juntos else 0
    df700["recibe_pension65"] = (df700[col_pension65].sum(axis=1) > 0).astype(int) if col_pension65 else 0
    df700["recibe_bono_fise"] = (df700[col_fise].sum(axis=1) > 0).astype(int) if col_fise else 0
    df700["recibe_wasi_mikuna"] = (df700[col_wasi].sum(axis=1) > 0).astype(int) if col_wasi else 0
    
    df700["recibe_subsidio_estatal"] = (
        (df700["recibe_juntos"] == 1) |
        (df700["recibe_pension65"] == 1) |
        (df700["recibe_bono_fise"] == 1) |
        (df700["recibe_wasi_mikuna"] == 1)
    ).astype(int)
    
    target_cols = [
        "CONGLOME", "VIVIENDA", "HOGAR",
        "recibe_juntos", "recibe_pension65", "recibe_bono_fise", "recibe_wasi_mikuna",
        "recibe_subsidio_estatal"
    ]
    return df700[target_cols].drop_duplicates(subset=id_cols)


def load_module_100(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa Módulo 100 (Vivienda, Colateral Predial, NBI y Conectividad).
    Recodifica colateral SUNARP, NBI Score (0 a 5), Área y Macrorregión.
    """
    logger.info(f"Cargando Módulo 100: {file_path}")
    header = pd.read_csv(file_path, sep=";", nrows=0, encoding="latin1").columns.tolist()
    clean_map = {c: c.replace("$", "_").strip() for c in header}
    
    needed = [
        "CONGLOME", "VIVIENDA", "HOGAR",
        "P106A", "P106B",
        "NBI1", "NBI2", "NBI3", "NBI4", "NBI5",
        "P1144", "P114B2", "P1144B2",
        "ESTRATO", "DOMINIO"
    ]
    target_raw = [c for c in header if clean_map[c] in needed]
    
    df100 = pd.read_csv(file_path, sep=";", usecols=target_raw, encoding="latin1", low_memory=False)
    df100.columns = [clean_map[c] for c in df100.columns]
    df100 = _normalize_keys(df100, ["CONGLOME", "VIVIENDA", "HOGAR"])
    
    # 1. Tenencia de título y colateral hipotecario / microfinanciero
    p106a = pd.to_numeric(df100.get("P106A", 0), errors="coerce").fillna(0)
    p106b = pd.to_numeric(df100.get("P106B", 0), errors="coerce").fillna(0)
    df100["tiene_colateral_propiedad"] = ((p106a == 1) & (p106b == 1)).astype(int)
    df100.loc[(df100["tiene_colateral_propiedad"] == 0) & (p106a == 1), "tiene_colateral_propiedad"] = 1
    
    # 2. Índice de Necesidades Básicas Insatisfechas (NBI 1 a 5)
    nbi_cols = ["NBI1", "NBI2", "NBI3", "NBI4", "NBI5"]
    for c in nbi_cols:
        if c in df100.columns:
            df100[c] = pd.to_numeric(df100[c], errors="coerce").fillna(0).astype(int)
        else:
            df100[c] = 0
    df100["nbi_score"] = df100[nbi_cols].sum(axis=1).clip(0, 5)
    
    # 3. Conectividad en el Hogar
    p1144 = pd.to_numeric(df100.get("P1144", 0), errors="coerce").fillna(0)
    df100["internet_hogar"] = (p1144 == 1).astype(int)
    
    p114b2_col = "P114B2" if "P114B2" in df100.columns else ("P1144B2" if "P1144B2" in df100.columns else None)
    if p114b2_col:
        p114b2 = pd.to_numeric(df100[p114b2_col], errors="coerce").fillna(0)
        df100["postpago_hogar"] = (p114b2 == 1).astype(int)
    else:
        df100["postpago_hogar"] = 0
        
    # 4. Geografía y Estratificación INEI
    estrato = pd.to_numeric(df100.get("ESTRATO", 1), errors="coerce").fillna(1).astype(int)
    dominio = pd.to_numeric(df100.get("DOMINIO", 1), errors="coerce").fillna(1).astype(int)
    
    df100["area"] = np.where(estrato <= 5, "Urbano", "Rural")
    df100["es_urbano"] = (df100["area"] == "Urbano").astype(int)
    
    macro_map = {
        1: "Costa", 2: "Costa", 3: "Costa",
        4: "Sierra", 5: "Sierra", 6: "Sierra",
        7: "Selva", 8: "Lima Metropolitana"
    }
    df100["dominio_macro"] = dominio.map(macro_map).fillna("Costa")
    
    estrato_map = {
        1: "Metrópoli (>=500k hab)",
        2: "Ciudad Intermedia Grande (100k-500k hab)",
        3: "Ciudad Mediana (50k-100k hab)",
        4: "Ciudad Menor (20k-50k hab)",
        5: "Pueblo Urbano (2k-20k hab)",
        6: "Asentamiento Rural Concentrado",
        7: "Área Rural Dispersa",
        8: "Área Rural Dispersa Especial"
    }
    df100["tamano_poblacional"] = estrato.map(estrato_map).fillna("Ciudad Mediana")
    
    keep_cols = [
        "CONGLOME", "VIVIENDA", "HOGAR",
        "tiene_colateral_propiedad", "nbi_score", "internet_hogar", "postpago_hogar",
        "area", "es_urbano", "dominio_macro", "tamano_poblacional"
    ]
    return df100[keep_cols].drop_duplicates(subset=["CONGLOME", "VIVIENDA", "HOGAR"])


def load_module_200(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa Módulo 200 (Miembros del Hogar y Demografía).
    Filtra universo adulto (18 a 65 años) y extrae sexo, edad, jefatura de hogar.
    """
    logger.info(f"Cargando Módulo 200: {file_path}")
    cols_to_use = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "P208A", "P207", "P203", "P209"]
    df200 = pd.read_csv(file_path, sep=";", usecols=cols_to_use, encoding="latin1", low_memory=False)
    df200 = _normalize_keys(df200, ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])
    
    df200["edad"] = pd.to_numeric(df200["P208A"], errors="coerce")
    df200 = df200[(df200["edad"] >= 18) & (df200["edad"] <= 65)].copy()
    
    p207 = pd.to_numeric(df200["P207"], errors="coerce").fillna(1)
    df200["es_mujer"] = (p207 == 2).astype(int)
    
    p203 = pd.to_numeric(df200["P203"], errors="coerce").fillna(2)
    df200["es_jefe_hogar"] = (p203 == 1).astype(int)
    
    df200["estado_civil"] = pd.to_numeric(df200["P209"], errors="coerce").fillna(6).astype(int)
    
    keep_cols = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "edad", "es_mujer", "es_jefe_hogar", "estado_civil"]
    return df200[keep_cols].drop_duplicates(subset=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])


def load_module_300(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa Módulo 300 (Educación, Lengua y Ecosistema Digital Individual).
    Construye variables de conectividad y habilidades digitales.
    """
    logger.info(f"Cargando Módulo 300: {file_path}")
    header = pd.read_csv(file_path, sep=";", nrows=0, encoding="latin1").columns.tolist()
    clean_map = {c: c.replace("$", "_").strip() for c in header}
    
    needed = [
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
        "P300A", "P301A", "P314A", "P314B1_9", "P316A1", "P316B", "P316_8"
    ]
    target_raw = [c for c in header if clean_map[c] in needed or (c.startswith("P316") and "8" in c and not "C" in c)]
    
    df300 = pd.read_csv(file_path, sep=";", usecols=target_raw, encoding="latin1", low_memory=False)
    df300.columns = [clean_map[c] for c in df300.columns]
    df300 = _normalize_keys(df300, ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])
    
    # Lengua materna
    p300a = pd.to_numeric(df300.get("P300A", 4), errors="coerce").fillna(4)
    df300["es_lengua_originaria"] = (p300a.isin([1, 2, 3, 10, 11, 12, 13, 14])).astype(int)
    
    # Nivel educativo ordinal
    df300["educacion_anios"] = pd.to_numeric(df300.get("P301A", 1), errors="coerce").fillna(1).astype(int)
    edu_desc_map = {
        1: "Sin Nivel / Inicial",
        2: "Primaria Incompleta",
        3: "Primaria Completa",
        4: "Secundaria Incompleta",
        5: "Secundaria Completa",
        6: "Superior No Universitaria Incompleta",
        7: "Superior No Universitaria Completa",
        8: "Superior Universitaria Incompleta",
        9: "Superior Universitaria Completa",
        10: "Maestría / Doctorado",
        11: "Básica Especial"
    }
    df300["nivel_educativo_desc"] = df300["educacion_anios"].map(edu_desc_map).fillna("Secundaria Completa")
    
    p314a = pd.to_numeric(df300.get("P314A", 2), errors="coerce").fillna(2)
    df300["usa_internet"] = (p314a == 1).astype(int)
    
    p316a1 = pd.to_numeric(df300.get("P316A1", 0), errors="coerce").fillna(0)
    df300["tiene_celular_propio"] = (p316a1 == 1).astype(int)
    
    p314b1_9 = pd.to_numeric(df300.get("P314B1_9", 0), errors="coerce").fillna(0)
    df300["tiene_plan_datos"] = ((p314b1_9 == 9) | (p314b1_9 == 1)).astype(int)
    
    p316b = pd.to_numeric(df300.get("P316B", 2), errors="coerce").fillna(2)
    df300["usa_computadora"] = (p316b == 1).astype(int)
    
    col_vta = [c for c in df300.columns if "P316" in c and "8" in c and not "C" in c]
    if col_vta:
        pvta = pd.to_numeric(df300[col_vta[0]], errors="coerce").fillna(0)
        df300["ventas_digitales"] = (pvta == 1).astype(int)
    else:
        df300["ventas_digitales"] = 0
        
    keep_cols = [
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
        "es_lengua_originaria", "educacion_anios", "nivel_educativo_desc",
        "usa_internet", "tiene_celular_propio", "tiene_plan_datos",
        "usa_computadora", "ventas_digitales"
    ]
    return df300[keep_cols].drop_duplicates(subset=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])


def load_module_400(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa Módulo 400 (Identidad Legal KYC y Aseguramiento en Salud).
    """
    logger.info(f"Cargando Módulo 400: {file_path}")
    cols_to_use = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "P401C", "P4191", "P4195"]
    df400 = pd.read_csv(file_path, sep=";", usecols=cols_to_use, encoding="latin1", low_memory=False)
    df400 = _normalize_keys(df400, ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])
    
    p401c = pd.to_numeric(df400["P401C"], errors="coerce").fillna(1)
    df400["tiene_dni"] = (p401c == 1).astype(int)
    df400["sin_dni_barrera"] = (p401c == 2).astype(int)
    
    p4191 = pd.to_numeric(df400["P4191"], errors="coerce").fillna(2)
    df400["seguro_essalud"] = (p4191 == 1).astype(int)
    
    p4195 = pd.to_numeric(df400["P4195"], errors="coerce").fillna(2)
    df400["seguro_sis"] = (p4195 == 1).astype(int)
    
    keep_cols = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "tiene_dni", "sin_dni_barrera", "seguro_essalud", "seguro_sis"]
    return df400[keep_cols].drop_duplicates(subset=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])


def load_module_500(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa Módulo 500 (Empleo, Ingresos, Inclusión Financiera y Factor FAC500A).
    """
    logger.info(f"Cargando Módulo 500: {file_path}")
    needed = [
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
        "FAC500A", "P558E1_6",
        "P507", "P510A1", "P511A", "P512A",
        "P524A1", "P530A",
        "P558A5", "P558G1", "P558G2", "P558G3", "P558G7"
    ]
    df500 = pd.read_csv(file_path, sep=";", usecols=needed, encoding="latin1", low_memory=False)
    df500 = _normalize_keys(df500, ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])
    
    # 1. Factor de expansión poblacional FAC500A
    if "FAC500A" in df500.columns:
        df500["FAC500A"] = (
            df500["FAC500A"]
            .astype(str)
            .str.replace(",", ".")
            .str.strip()
        )
        df500["FAC500A"] = pd.to_numeric(df500["FAC500A"], errors="coerce").fillna(1.0)
    else:
        df500["FAC500A"] = 1.0
        
    # 2. Definición formal de Bancarización:
    p558e1_6 = pd.to_numeric(df500.get("P558E1_6", 0), errors="coerce").fillna(0)
    df500["no_bancarizado_flag"] = (p558e1_6 == 6).astype(int)
    df500["IS_BANCARIZADO"] = (p558e1_6 != 6).astype(int)
    
    # 3. Estructura ocupacional
    df500["categoria_ocupacional"] = pd.to_numeric(df500.get("P507", np.nan), errors="coerce")
    cat_desc_map = {
        1: "Empleador / Patrono",
        2: "Trabajador Independiente",
        3: "Empleado Asalariado",
        4: "Obrero",
        5: "Trabajador Familiar No Remunerado (TFNR)",
        6: "Trabajador del Hogar",
        7: "Otro"
    }
    df500["categoria_ocupacional_desc"] = df500["categoria_ocupacional"].map(cat_desc_map).fillna("Independiente / No Calificado")
    
    p510a1 = pd.to_numeric(df500.get("P510A1", 3), errors="coerce").fillna(3)
    df500["tiene_ruc"] = (p510a1.isin([1, 2])).astype(int)
    
    df500["tamano_empresa"] = pd.to_numeric(df500.get("P512A", np.nan), errors="coerce")
    df500["tipo_contrato"] = pd.to_numeric(df500.get("P511A", np.nan), errors="coerce")
    
    df500["informalidad"] = (
        (df500["tiene_ruc"] == 0) |
        (df500["categoria_ocupacional"] == 5) |
        (df500["tipo_contrato"].isin([7, np.nan]) & df500["categoria_ocupacional"].isin([3, 4]))
    ).astype(int)
    
    df500["empleo_vulnerable"] = (
        ((df500["categoria_ocupacional"] == 2) & (df500["tiene_ruc"] == 0)) |
        (df500["categoria_ocupacional"].isin([5, 6]))
    ).astype(int)
    
    # 4. Ingresos laborales brutos (previo a imputación econométrica)
    df500["ingreso_dependiente"] = pd.to_numeric(df500.get("P524A1", np.nan), errors="coerce")
    df500["ganancia_independiente"] = pd.to_numeric(df500.get("P530A", np.nan), errors="coerce")
    
    # Centinelas ENAHO (999999) a NaN
    df500.loc[df500["ingreso_dependiente"] >= 999990, "ingreso_dependiente"] = np.nan
    df500.loc[df500["ganancia_independiente"] >= 999990, "ganancia_independiente"] = np.nan
    
    # 5. Conductas financieras informales
    p558a5 = pd.to_numeric(df500.get("P558A5", 0), errors="coerce").fillna(0)
    df500["sin_pension"] = (p558a5 == 5).astype(int)
    
    p558g1 = pd.to_numeric(df500.get("P558G1", 0), errors="coerce").fillna(0)
    df500["ahorro_juntas"] = (p558g1 == 1).astype(int)
    
    p558g2 = pd.to_numeric(df500.get("P558G2", 0), errors="coerce").fillna(0)
    df500["ahorro_familiares"] = (p558g2.isin([1, 2])).astype(int)
    
    p558g3 = pd.to_numeric(df500.get("P558G3", 0), errors="coerce").fillna(0)
    df500["ahorro_casa"] = (p558g3.isin([1, 3])).astype(int)
    
    p558g7 = pd.to_numeric(df500.get("P558G7", 0), errors="coerce").fillna(0)
    df500["prestamo_informal"] = (p558g7.isin([1, 7])).astype(int)
    
    keep_cols = [
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
        "FAC500A", "IS_BANCARIZADO", "no_bancarizado_flag",
        "categoria_ocupacional", "categoria_ocupacional_desc",
        "tiene_ruc", "tamano_empresa", "tipo_contrato",
        "informalidad", "empleo_vulnerable",
        "ingreso_dependiente", "ganancia_independiente",
        "sin_pension", "ahorro_juntas", "ahorro_familiares", "ahorro_casa", "prestamo_informal"
    ]
    return df500[keep_cols].drop_duplicates(subset=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"])


def integrate_enaho_2025_datasets(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función maestra de integración relacional de los módulos de ENAHO 2025.
    Retorna:
    - df_total_adults: Panel completo de adultos (18 a 65 años) para Modelado Look-Alike.
    - df_unbanked: Población no bancarizada objetivo para Modelo LCA e IVF.
    """
    data_dir = config["paths"]["data_dir"]
    files = config["files"]
    
    path_100 = os.path.join(data_dir, files["mod_100"])
    path_200 = os.path.join(data_dir, files["mod_200"])
    path_300 = os.path.join(data_dir, files["mod_300"])
    path_400 = os.path.join(data_dir, files["mod_400"])
    path_500 = os.path.join(data_dir, files["mod_500"])
    path_602 = os.path.join(data_dir, files["mod_602"])
    path_700 = os.path.join(data_dir, files["mod_700"])
    
    logger.info("=== INICIANDO INTEGRACIÓN MULTI-NIVEL ENAHO 2025 ===")
    
    # 1. Carga de módulos de hogar y agregados
    df_food_aid = load_and_aggregate_food_aid(path_602)
    df_subsidies = load_and_aggregate_subsidies(path_700)
    df_100 = load_module_100(path_100)
    
    id_h = ["CONGLOME", "VIVIENDA", "HOGAR"]
    df_hogar = pd.merge(df_100, df_food_aid, on=id_h, how="left")
    df_hogar["acude_olla_comedor"] = df_hogar["acude_olla_comedor"].fillna(0).astype(int)
    
    df_hogar = pd.merge(df_hogar, df_subsidies, on=id_h, how="left")
    for sub_col in ["recibe_juntos", "recibe_pension65", "recibe_bono_fise", "recibe_wasi_mikuna", "recibe_subsidio_estatal"]:
        df_hogar[sub_col] = df_hogar[sub_col].fillna(0).astype(int)
        
    logger.info(f"Nivel Hogar consolidado: {len(df_hogar):,} hogares.")
    
    # 2. Carga de módulos de individuos
    df_200 = load_module_200(path_200)
    df_300 = load_module_300(path_300)
    df_400 = load_module_400(path_400)
    df_500 = load_module_500(path_500)
    
    id_p = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]
    
    df_individuo = pd.merge(df_200, df_500, on=id_p, how="inner")
    df_individuo = pd.merge(df_individuo, df_300, on=id_p, how="left")
    df_individuo = pd.merge(df_individuo, df_400, on=id_p, how="left")
    
    # 3. Fusión Hogar + Individuo
    df_total = pd.merge(df_individuo, df_hogar, on=id_h, how="inner")
    
    # 4. Construcción del nuevo Índice de Acceso Digital (0 a 5)
    c1 = ((df_total["internet_hogar"] == 1) | (df_total["postpago_hogar"] == 1)).astype(int)
    c2 = (df_total["tiene_celular_propio"] == 1).astype(int)
    c3 = (df_total["tiene_plan_datos"] == 1).astype(int)
    c4 = (df_total["usa_internet"] == 1).astype(int)
    c5 = ((df_total["usa_computadora"] == 1) | (df_total["ventas_digitales"] == 1)).astype(int)
    df_total["acceso_digital"] = (c1 + c2 + c3 + c4 + c5).clip(0, 5)
    
    for b_col in ["tiene_dni", "sin_dni_barrera", "seguro_essalud", "seguro_sis", "es_lengua_originaria"]:
        if b_col in df_total.columns:
            df_total[b_col] = df_total[b_col].fillna(0).astype(int)
            
    logger.info(f"Panel Total Adulto integrado: {len(df_total):,} registros.")
    
    df_unbanked = df_total[df_total["no_bancarizado_flag"] == 1].copy()
    logger.info(f"Población Objetivo No Bancarizada: {len(df_unbanked):,} individuos (Tasa de Exclusión: {len(df_unbanked)/len(df_total)*100:.1f}%).")
    
    return df_total, df_unbanked


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    df_tot, df_unb = integrate_enaho_2025_datasets(cfg)
    print("Muestra exitosa:")
    print("df_total shape:", df_tot.shape)
    print("df_unbanked shape:", df_unb.shape)
    print("Distribución IS_BANCARIZADO:\n", df_tot["IS_BANCARIZADO"].value_counts())
