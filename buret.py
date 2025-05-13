#!/usr/bin/env python3
"""buret_analysis_simple.py  
Versión minimalista (pandas + numpy + opcional SciPy) adaptada a las **variables enteras** del protocolo BURET.

### Columnas esperadas (tipo **int** en el CSV)
| Columna | Descripción |
|---------|-------------|
| `edad`                       | Edad del participante |
| `sexo`                       | 0 = mujer, 1 = hombre |
| `uso_redes`                  | Puntuación total en escala de uso problemático de redes sociales |
| `burnout`                    | Puntuación total MBI (0‑48) |
| `factores_psicosociales`     | Puntaje total COPSOQ‑ISTAS21 (0‑64) |

### Clasificaciones incorporadas
| Variable | Rangos |
|----------|--------|
| **Burnout (MBI)** | 0‑17 → **bajo** · 18‑29 → **moderado** · ≥30 → **alto** |
| **COPSOQ**        | 0‑16 → **verde** · 24‑32 → **amarillo** · 40‑64 → **rojo** |

### Salida del script
1. Estadísticos descriptivos numéricos (`edad`, `uso_redes`, `burnout`, `factores_psicosociales`)
2. Frecuencias para `sexo`, `nivel_burnout` y `nivel_copsoq`
3. Matriz de correlación de Pearson (numéricas)
4. Comparación *High vs Low uso_redes* (corte en mediana) sobre **burnout**
   - media ± DE por grupo
   - tamaño de efecto (*Cohen d*)
   - *t* de Welch y *p* (requiere SciPy)

> **Nota**: si necesitas análisis de regresión o moderación, será mejor añadir `statsmodels`.
"""

from __future__ import annotations
import sys
import pandas as pd
import numpy as np

# SciPy se usa solo para el p‑value; si no existe, se continua sin él.
try:
    from scipy import stats  # type: ignore
except ImportError:
    stats = None

# ---------------------------------------------------------------------------
# Clasificación de escalas
# ---------------------------------------------------------------------------

def classify_burnout(score: int) -> str:
    if score <= 17:
        return "bajo"
    elif 18 <= score <= 29:
        return "moderado"
    else:  # >=30
        return "alto"


def classify_copsoq(score: int) -> str:
    if 0 <= score <= 16:
        return "verde"
    elif 24 <= score <= 32:
        return "amarillo"
    elif 40 <= score <= 64:
        return "rojo"
    else:
        return "fuera_rango"  # valores intermedios que no caen en los tramos oficiales

# ---------------------------------------------------------------------------
# Funciones auxiliares de reporte
# ---------------------------------------------------------------------------

def descriptive_numeric(df: pd.DataFrame, cols: list[str]):
    print("\n-- Estadísticos descriptivos (numéricos) --")
    tbl = df[cols].agg(["count", "mean", "std", "min", "max"]).T.round(2)
    print(tbl)


def freq_table(series: pd.Series, nombre: str):
    frec = series.value_counts(dropna=False)
    rel = series.value_counts(normalize=True, dropna=False).round(3) * 100
    tbl = pd.DataFrame({"n": frec.astype(int), "%": rel})
    print(f"\n-- Distribución de '{nombre}' --")
    print(tbl)


def correlations(df: pd.DataFrame, cols: list[str]):
    print("\n-- Correlaciones de Pearson --")
    print(df[cols].corr().round(3))


def median_split(df: pd.DataFrame, var: str):
    med = df[var].median()
    hi = df[df[var] > med]
    lo = df[df[var] <= med]
    return med, hi, lo


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_sd


def compare_high_low(df: pd.DataFrame, outcome: str, split_var: str):
    med, hi, lo = median_split(df, split_var)
    print(f"\n-- Comparación High vs Low en '{split_var}' (mediana = {med}) --")
    print(f"High (n={len(hi)}): {outcome} = {hi[outcome].mean():.2f} ± {hi[outcome].std():.2f}")
    print(f"Low  (n={len(lo)}): {outcome} = {lo[outcome].mean():.2f} ± {lo[outcome].std():.2f}")

    d = cohens_d(hi[outcome].values, lo[outcome].values)
    print(f"Cohen d = {d:.3f}")

    if stats is not None:
        t_stat, p_val = stats.ttest_ind(hi[outcome], lo[outcome], equal_var=False)
        print(f"Welch t = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        print("[Sin SciPy] Instala scipy para obtener p-value")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Uso: python buret_analysis_simple.py data.csv")
        sys.exit(0)

    df = pd.read_csv(sys.argv[1])

    # Conversión forzada a int (por si vienen como float)
    int_cols = ["edad", "sexo", "uso_redes", "burnout", "factores_psicosociales"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Crear columnas de clasificación
    df["nivel_burnout"] = df["burnout"].apply(classify_burnout)
    df["nivel_copsoq"] = df["factores_psicosociales"].apply(classify_copsoq)

    numeric_vars = ["edad", "uso_redes", "burnout", "factores_psicosociales"]

    # --- Reportes ---
    descriptive_numeric(df, numeric_vars)
    freq_table(df["sexo"], "sexo (0=mujer,1=hombre)")
    freq_table(df["nivel_burnout"], "nivel_burnout")
    freq_table(df["nivel_copsoq"], "nivel_copsoq")

    correlations(df, numeric_vars)
    compare_high_low(df, outcome="burnout", split_var="uso_redes")


if __name__ == "__main__":
    main()
