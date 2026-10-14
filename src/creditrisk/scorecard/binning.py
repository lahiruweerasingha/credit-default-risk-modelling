"""Weight-of-evidence binning with monotonic bad rates.

Convention: WoE = ln(%good / %bad), so a higher WoE means lower risk.

Numeric features
    1. Pre-bin on quantiles (or on each distinct value when there are few).
    2. Pick the trend direction from the Spearman correlation between x and the bad flag.
    3. Merge adjacent bins until bad rates are strictly monotonic in that direction.
    4. Merge bins smaller than `min_bin_share` into the adjacent bin with the closest bad rate
       (merging adjacent bins of a monotonic sequence keeps it monotonic), but never below two
       bins, so a rare high-risk state is not absorbed into the majority bin.
    Missing values get their own bin when there are at least `min_missing_count` of them,
    otherwise a neutral WoE of 0.

Categorical features
    Categories below `min_bin_share` are pooled into "__OTHER__"; missing is its own category.
    Unseen categories at scoring time fall into "__OTHER__" (or WoE 0 if no such bin exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MISSING = "__MISSING__"
OTHER = "__OTHER__"


@dataclass
class WoEBinner:
    name: str
    kind: str = "numeric"
    max_prebins: int = 20
    min_bin_share: float = 0.05
    min_missing_count: int = 50
    smoothing: float = 0.5

    edges: list[float] = field(default_factory=list)  # interior cut points, bins are (a, b]
    trend: str | None = None
    category_map: dict[str, int] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    iv: float = 0.0

    # ------------------------------------------------------------------ fitting
    def fit(self, x: pd.Series, y: np.ndarray) -> WoEBinner:
        y = np.asarray(y, dtype=int)
        if self.kind == "numeric":
            self._fit_numeric(pd.to_numeric(x, errors="coerce").to_numpy(dtype=float), y)
        else:
            self._fit_categorical(x, y)
        return self

    def _fit_numeric(self, x: np.ndarray, y: np.ndarray) -> None:
        present = ~np.isnan(x)
        xv, yv = x[present], y[present]
        uniques = np.unique(xv)
        if len(uniques) <= 1:
            edges: list[float] = []
        elif len(uniques) <= self.max_prebins:
            edges = list(uniques[:-1])
        else:
            qs = np.linspace(0, 1, self.max_prebins + 1)[1:-1]
            edges = sorted(set(np.quantile(xv, qs, method="lower").tolist()))
            edges = [e for e in edges if e < uniques[-1]]

        if len(uniques) > 1:
            rho = spearmanr(xv, yv).statistic
            self.trend = "increasing" if (np.isnan(rho) or rho >= 0) else "decreasing"
            edges = self._merge_monotonic(xv, yv, edges)
            edges = self._merge_small(xv, yv, edges, n_total=len(x))
        self.edges = [float(e) for e in edges]
        self._build_table(self._numeric_index(x), y, n_bins=len(self.edges) + 1, labels=None)

    def _counts(self, xv: np.ndarray, yv: np.ndarray, edges: list[float]):
        idx = np.searchsorted(np.asarray(edges), xv, side="left")
        n = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        bad = np.bincount(idx, weights=yv, minlength=len(edges) + 1)
        return n, bad

    def _violates(self, rates: np.ndarray) -> int | None:
        for i in range(len(rates) - 1):
            if self.trend == "increasing" and rates[i + 1] <= rates[i]:
                return i
            if self.trend == "decreasing" and rates[i + 1] >= rates[i]:
                return i
        return None

    def _merge_monotonic(self, xv, yv, edges: list[float]) -> list[float]:
        edges = list(edges)
        while edges:
            n, bad = self._counts(xv, yv, edges)
            keep = n > 0
            if not keep.all():  # drop empty bins first
                if not keep[-1]:
                    edges.pop()
                else:
                    edges = [e for e, k in zip(edges, keep[:-1], strict=True) if k]
                continue
            i = self._violates(bad / n)
            if i is None:
                break
            del edges[i]  # merge bin i with bin i+1
        return edges

    def _merge_small(self, xv, yv, edges: list[float], n_total: int) -> list[float]:
        edges = list(edges)
        # Never merge below two bins: a rare but highly informative state (e.g. "any
        # delinquency", ~3% of loans) must keep its own bin rather than vanish into the majority.
        while len(edges) > 1:
            n, bad = self._counts(xv, yv, edges)
            share = n / n_total
            i = int(np.argmin(share))
            if share[i] >= self.min_bin_share:
                break
            rates = bad / np.maximum(n, 1)
            if i == 0:
                j = 1
            elif i == len(n) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(rates[i - 1] - rates[i]) <= abs(rates[i + 1] - rates[i]) else i + 1
            del edges[min(i, j)]  # removing the edge between bins i and j merges them
        return edges

    def _fit_categorical(self, x: pd.Series, y: np.ndarray) -> None:
        xs = _as_category_strings(x)
        counts = xs[xs != MISSING].value_counts()
        n_total = len(xs)
        big = [c for c, n in counts.items() if n / n_total >= self.min_bin_share]
        small = [c for c in counts.index if c not in big]
        labels = list(big) + ([OTHER] if small else [])
        self.category_map = {c: labels.index(c) for c in big}
        self.category_map.update({c: labels.index(OTHER) for c in small})
        if small:
            self.category_map[OTHER] = labels.index(OTHER)
        idx = self._categorical_index(xs, n_known=len(labels))
        self._build_table(idx, y, n_bins=len(labels), labels=labels)

    # ------------------------------------------------------------------ indexing
    def _numeric_index(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(np.asarray(self.edges), x, side="left")
        idx[np.isnan(x)] = len(self.edges) + 1  # missing bin is the last row
        return idx

    def _categorical_index(self, xs: pd.Series, n_known: int) -> np.ndarray:
        other = self.category_map.get(OTHER, -1)
        mapped = xs.map(self.category_map)
        idx = mapped.fillna(other).to_numpy(dtype=int)
        idx[(xs == MISSING).to_numpy()] = n_known  # missing bin is the last row
        idx[idx == -1] = n_known + 1  # unseen and no OTHER bin -> neutral row
        return idx

    def bin_index(self, x) -> np.ndarray:
        if self.kind == "numeric":
            return self._numeric_index(pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float))
        n_known = len(self.table) - 2
        return self._categorical_index(_as_category_strings(pd.Series(x)), n_known)

    # ------------------------------------------------------------------ statistics
    def _build_table(self, idx: np.ndarray, y: np.ndarray, n_bins: int, labels) -> None:
        n_rows = n_bins + 2  # regular bins + missing + neutral fallback
        n = np.bincount(idx, minlength=n_rows)[:n_rows].astype(float)
        bad = np.bincount(idx, weights=y, minlength=n_rows)[:n_rows]
        good = n - bad
        total_bad, total_good = max(bad.sum(), 1.0), max(good.sum(), 1.0)
        s = self.smoothing
        woe = np.log(((good + s) / total_good) / ((bad + s) / total_bad))
        missing_row = n_bins
        if n[missing_row] < self.min_missing_count:
            woe[missing_row] = 0.0
        woe[n_bins + 1] = 0.0  # neutral fallback row
        iv_parts = (good / total_good - bad / total_bad) * woe
        if labels is None:
            labels = _interval_labels(self.edges)
        table = pd.DataFrame(
            {
                "bin": list(labels) + ["Missing", "Unseen"],
                "count": n.astype(int),
                "bads": bad.astype(int),
                "bad_rate": np.divide(bad, n, out=np.full_like(bad, np.nan), where=n > 0),
                "woe": woe,
                "iv": iv_parts,
            }
        )
        self.table = table
        self.iv = float(iv_parts.sum())

    def transform(self, x) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.bin_index(x)]

    def bin_labels(self, x) -> np.ndarray:
        return self.table["bin"].to_numpy()[self.bin_index(x)]

    def is_monotonic(self) -> bool:
        """Bad rates of the regular (non-missing) numeric bins are strictly monotonic."""
        if self.kind != "numeric" or len(self.edges) == 0:
            return True
        rates = self.table["bad_rate"].to_numpy()[: len(self.edges) + 1]
        diffs = np.diff(rates)
        return bool((diffs > 0).all() or (diffs < 0).all())


def _as_category_strings(x: pd.Series) -> pd.Series:
    xs = pd.Series(x).astype("object")
    return xs.where(xs.notna(), MISSING).astype(str)


def _interval_labels(edges: list[float]) -> list[str]:
    if not edges:
        return ["All"]
    fmt = "{:.4g}".format
    labels = [f"<= {fmt(edges[0])}"]
    labels += [f"({fmt(a)}, {fmt(b)}]" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    labels.append(f"> {fmt(edges[-1])}")
    return labels
