from __future__ import annotations

from typing import Literal, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


VarType = Literal["continua", "discreta"]


def _clean_input(x: np.ndarray) -> np.ndarray:
    """
    Limpia el vector de entrada:
    - Asegura que sea 1D.
    - Elimina NaN.
    """
    arr = np.asarray(x).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size < 5:
        raise ValueError(
            "Se requieren al menos 5 observaciones para ajustar distribuciones."
        )
    return arr


def fit_one_distribution(
    x: np.ndarray,
    dist_name: str,
    kind: Literal["continuous", "discrete"],
) -> Dict[str, Any]:
    x = _clean_input(x)

    if kind == "continuous":
        dist = getattr(stats, dist_name)
        params = dist.fit(x)
        logpdf = dist.logpdf(x, *params)
        loglik = float(np.sum(logpdf))
        D, p_gof = stats.kstest(x, dist_name, args=params)
        gof_test = "KS"
        gof_stat = float(D)

    else:
        # Discretas: ajustamos con fit (cuando exista) + Chi-cuadrado
        x_int = np.asarray(x).astype(int)
        if not np.allclose(x, x_int):
            x_int = np.round(x).astype(int)

        dist = getattr(stats, dist_name)

        if dist_name == "poisson":
            # Puedes mantener este caso cerrado si quieres el MLE "manual"
            lam_hat = float(x_int.mean())
            params = (lam_hat,)
        else:
            # Ajuste genérico
            params = dist.fit(x_int)

        # Log-verosimilitud
        logpmf = dist.logpmf(x_int, *params)
        loglik = float(np.sum(logpmf))

        # Bondad de ajuste tipo Chi-cuadrado
        gof_test = "Chi2"
        try:
            values, counts = np.unique(x_int, return_counts=True)
            pmf = dist.pmf(values, *params)
            expected = pmf * x_int.size

            mask = expected > 1e-8
            if mask.sum() < 2:
                raise RuntimeError("Muy pocos valores esperados > 0 para Chi-cuadrado.")

            chi2, p_gof = stats.chisquare(
                f_obs=counts[mask],
                f_exp=expected[mask],
            )
            gof_stat = float(chi2)
            p_gof = float(p_gof)
        except Exception:
            gof_stat = float("nan")
            p_gof = float("nan")

    k = len(params)
    n = x.size
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        "dist": dist_name,
        "kind": kind,
        "params": params,
        "loglik": loglik,
        "AIC": float(aic),
        "BIC": float(bic),
        "GOF_test": gof_test,
        "GOF_stat": gof_stat,
        "GOF_p": p_gof,
    }


def _default_candidates(var_type: VarType) -> List[str]:
    """
    Devuelve una lista de distribuciones candidatas por defecto
    según el tipo de variable.
    """
    if var_type == "continua":
        return [
            "norm",
            "t",
            "expon",
            "gamma",
            "weibull_max",
            "weibull_min",
            "lognorm",
            "chi2",
            "f",
            "beta",
            "cauchy",
            "pareto",
            "logistic",
            "gumbel_r",
            "gumbel_l",
            "triang",
            "uniform",
            "laplace",
            "rayleigh",
        ]
    elif var_type == "discreta":
        return [
            "poisson",
            "bernoulli",
            "binom",
            "geom",
            "hypergeom",
            "nbinom",
            "randint",
        ]
    else:
        raise ValueError("var_type debe ser 'continua' o 'discreta'.")


def compare_distributions(
    x: np.ndarray,
    var_type: VarType,
    candidates: List[str] | None = None,
) -> pd.DataFrame:
    """
    Ajusta automáticamente varias distribuciones a un vector de datos x
    según el tipo de variable (continua o discreta), y devuelve un
    resumen ordenado por AIC (mejor ajuste arriba).
    """
    x = _clean_input(x)

    vt = var_type.lower()
    if vt not in ("continua", "discreta"):
        raise ValueError("var_type debe ser 'continua' o 'discreta'.")

    if candidates is None:
        candidates = _default_candidates(vt)  # type: ignore[arg-type]

    kind = "continuous" if vt == "continua" else "discrete"

    results: List[Dict[str, Any]] = []
    for name in candidates:
        try:
            res = fit_one_distribution(x, name, kind=kind)
            results.append(res)
        except Exception as e:
            # Si quieres ver los errores, descomenta:
            print(f"Fallo al ajustar {name}: {e}")
            continue

    if not results:
        raise ValueError("No se pudo ajustar ninguna distribución a los datos.")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("AIC").reset_index(drop=True)
    return df_sorted


if __name__ == "__main__":
    rng = np.random.default_rng(2025)

    # Continua: normal
    x_cont = rng.normal(loc=50, scale=5, size=300)
    print("=== Variable continua ===")
    res_cont = compare_distributions(x_cont, var_type="continua")
    print(res_cont[["dist", "AIC", "BIC", "GOF_test", "GOF_p"]])

    # Discreta: Poisson
    x_disc = rng.poisson(lam=3, size=3000)
    print("\n=== Variable discreta ===")
    res_disc = compare_distributions(x_disc, var_type="discreta")
    print(res_disc[["dist", "AIC", "BIC", "GOF_test", "GOF_p"]])
