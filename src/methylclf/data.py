"""The only place the beta-value store is read.

Why one module: the Zarr store holds ALL samples (including the ones that
failed QC) and ALL probes (including the filtered ones). Which rows and
columns a model may see is decided by two small tables:

  fold table   (results/splits/folds_seed42_qc.tsv)  geo_accession -> zarr_row
  probe table  (results/probes/probes_kept.tsv)      Probe_ID      -> col

If every script did its own `betas[i, j]`, one of them would eventually use a
fold-table position as a store row and silently read the wrong sample. Here
the lookup is done once, and checked against the store's own index files.

Callers ask by NAME (sample IDs, probe IDs) and never see a row number.
A sample that failed QC is not in the fold table, so it cannot be loaded.

Usage:
    store = BetaStore.open("data/betas/zarr/GSE90496.zarr",
                           "results/splits/folds_seed42_qc.tsv",
                           "results/probes/probes_kept.tsv")
    X = store.load(train_ids)                # samples x 381,355, float32, NaN = missing
    X = store.load(train_ids, probe_ids)     # only these probes, in this order
    st = store.probe_stats(train_ids)        # per-probe missingness/variance, streamed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """An assumption about the store or its tables is broken."""


def _fail(msg: str):
    raise DataError(f"ERROR: {msg}")


def _read_tsv(path: Path, required: list[str]) -> pd.DataFrame:
    if not Path(path).is_file():
        _fail(f"file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    return df


def _as_int(df: pd.DataFrame, col: str, path: Path) -> np.ndarray:
    try:
        return df[col].astype(np.int64).to_numpy()
    except (ValueError, TypeError):
        _fail(f"{path}: column '{col}' is not all integers")


class BetaStore:
    """Name-based, read-only access to one cohort's beta values."""

    def __init__(self, betas, samples: pd.DataFrame, probes: pd.DataFrame, name: str = ""):
        # `betas` is anything with .shape, .chunks and numpy-style slicing
        # (a zarr array in real use). Only plain slices are ever requested.
        self._betas = betas
        self.samples = samples  # fold table; column zarr_row is int
        self.probes = probes  # kept probes; column col is int
        self.name = name
        self._row_of = dict(zip(samples["geo_accession"], samples["zarr_row"]))
        self._col_of = dict(zip(probes["Probe_ID"], probes["col"]))

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, zarr_path, folds_path, probes_path,
             sample_index_path=None, probe_index_path=None) -> "BetaStore":
        """Open the store and check both lookup tables against its index files.

        The index files written by build_zarr.py sit next to the store:
        <name>.samples.tsv (row, geo_accession, ...) and <name>.probes.tsv
        (col, Probe_ID). They are the store's own record of what each row and
        column is; the fold and probe tables must agree with them.
        """
        import zarr  # imported here so the module can be imported without it

        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            _fail(f"store not found: {zarr_path}")
        stem = zarr_path.name.removesuffix(".zarr")
        sample_index_path = Path(sample_index_path or zarr_path.with_name(f"{stem}.samples.tsv"))
        probe_index_path = Path(probe_index_path or zarr_path.with_name(f"{stem}.probes.tsv"))

        betas = zarr.open_group(str(zarr_path), mode="r")["betas"]
        n_rows, n_cols = betas.shape

        # ---- samples: fold table vs the store's sample index
        folds = _read_tsv(folds_path, ["geo_accession", "zarr_row"])
        folds["zarr_row"] = _as_int(folds, "zarr_row", folds_path)
        if folds["geo_accession"].duplicated().any():
            dup = folds.loc[folds["geo_accession"].duplicated(), "geo_accession"].iloc[0]
            _fail(f"{folds_path}: sample {dup} appears more than once")
        if folds["zarr_row"].duplicated().any():
            _fail(f"{folds_path}: two samples point at the same zarr_row")

        sidx = _read_tsv(sample_index_path, ["row", "geo_accession"])
        sidx["row"] = _as_int(sidx, "row", sample_index_path)
        if len(sidx) != n_rows or not np.array_equal(sidx["row"], np.arange(n_rows)):
            _fail(f"{sample_index_path}: expected rows 0..{n_rows - 1} in order, "
                  f"to match the store's {n_rows} rows")
        true_row = dict(zip(sidx["geo_accession"], sidx["row"]))
        for gsm, row in zip(folds["geo_accession"], folds["zarr_row"]):
            if gsm not in true_row:
                _fail(f"{folds_path}: sample {gsm} is not in {sample_index_path}")
            if true_row[gsm] != row:
                _fail(f"{folds_path}: sample {gsm} has zarr_row {row}, but the store "
                      f"index says row {true_row[gsm]}")

        # ---- probes: kept-probe table vs the store's probe index
        probes = _read_tsv(probes_path, ["col", "Probe_ID"])
        probes["col"] = _as_int(probes, "col", probes_path)
        if probes["Probe_ID"].duplicated().any():
            _fail(f"{probes_path}: duplicated Probe_ID")
        pidx = _read_tsv(probe_index_path, ["col", "Probe_ID"])
        pidx["col"] = _as_int(pidx, "col", probe_index_path)
        if len(pidx) != n_cols or not np.array_equal(pidx["col"], np.arange(n_cols)):
            _fail(f"{probe_index_path}: expected cols 0..{n_cols - 1} in order, "
                  f"to match the store's {n_cols} columns")
        if probes["col"].min() < 0 or probes["col"].max() >= n_cols:
            _fail(f"{probes_path}: col outside 0..{n_cols - 1}")
        expected = pidx["Probe_ID"].to_numpy()[probes["col"].to_numpy()]
        bad = np.flatnonzero(expected != probes["Probe_ID"].to_numpy())
        if bad.size:
            i = bad[0]
            _fail(f"{probes_path}: probe {probes['Probe_ID'].iloc[i]} has col "
                  f"{probes['col'].iloc[i]}, but the store has {expected[i]} there")

        return cls(betas, folds.reset_index(drop=True), probes.reset_index(drop=True), stem)

    # --------------------------------------------------------------- lookups
    @property
    def sample_ids(self) -> np.ndarray:
        return self.samples["geo_accession"].to_numpy()

    @property
    def probe_ids(self) -> np.ndarray:
        return self.probes["Probe_ID"].to_numpy()

    def _rows(self, sample_ids) -> np.ndarray:
        ids = list(sample_ids)
        if len(ids) == 0:
            _fail("no sample IDs given")
        if len(set(ids)) != len(ids):
            _fail("the same sample ID was requested more than once")
        unknown = [s for s in ids if s not in self._row_of]
        if unknown:
            _fail(f"{len(unknown)} sample(s) are not in the fold table (failed QC, or "
                  f"another cohort), first: {unknown[0]}")
        return np.array([self._row_of[s] for s in ids], dtype=np.int64)

    def _cols(self, probe_ids) -> np.ndarray:
        if probe_ids is None:
            return self.probes["col"].to_numpy()
        ids = list(probe_ids)
        if len(ids) == 0:
            _fail("no probe IDs given")
        if len(set(ids)) != len(ids):
            _fail("the same probe ID was requested more than once")
        unknown = [p for p in ids if p not in self._col_of]
        if unknown:
            _fail(f"{len(unknown)} probe(s) are not in the kept-probe table, "
                  f"first: {unknown[0]}")
        return np.array([self._col_of[p] for p in ids], dtype=np.int64)

    # --------------------------------------------------------------- reading
    def _iter_blocks(self, rows: np.ndarray, cols: np.ndarray):
        """Yield (positions, block): block is samples x len(cols), float32.

        `positions` are indexes into the caller's `rows`. The store is read
        one chunk-height of rows at a time with plain slices (fast, and it
        keeps peak memory at one block), then rows and columns are picked
        in numpy. Columns outside min(cols)..max(cols) are never read.
        """
        step = int(self._betas.chunks[0])
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        rel_cols = cols - c0
        order = np.argsort(rows, kind="stable")
        block_of = rows[order] // step
        for b in np.unique(block_of):
            pos = order[block_of == b]
            r = rows[pos]
            lo, hi = int(r.min()), int(r.max()) + 1
            raw = np.asarray(self._betas[lo:hi, c0:c1])
            yield pos, raw[r - lo][:, rel_cols].astype(np.float32, copy=False)

    def load(self, sample_ids, probe_ids=None) -> np.ndarray:
        """Beta values, rows in the order of `sample_ids`, columns in the order
        of `probe_ids` (default: every kept probe, in probe-table order).
        float32; NaN means missing."""
        rows, cols = self._rows(sample_ids), self._cols(probe_ids)
        out = np.empty((rows.size, cols.size), dtype=np.float32)
        for pos, block in self._iter_blocks(rows, cols):
            out[pos] = block
        return out

    def probe_stats(self, sample_ids, probe_ids=None) -> pd.DataFrame:
        """Per-probe statistics over `sample_ids` ONLY, without holding the
        matrix in memory.

        Returns one row per probe: n_obs, frac_missing, mean, var (ddof=1,
        over observed values). Because the answer depends on which samples
        are passed, pass training-fold samples only when the result feeds a
        model.
        """
        rows, cols = self._rows(sample_ids), self._cols(probe_ids)
        n = np.zeros(cols.size, dtype=np.int64)
        s = np.zeros(cols.size, dtype=np.float64)
        ss = np.zeros(cols.size, dtype=np.float64)
        for _, block in self._iter_blocks(rows, cols):
            ok = ~np.isnan(block)
            x = np.where(ok, block, 0.0).astype(np.float64)
            n += ok.sum(axis=0)
            s += x.sum(axis=0)
            ss += (x * x).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
            var = np.where(n > 1, (ss - n * mean**2) / (n - 1), np.nan)
        ids = self.probe_ids if probe_ids is None else np.array(list(probe_ids))
        return pd.DataFrame({
            "Probe_ID": ids,
            "n_obs": n,
            "frac_missing": 1.0 - n / rows.size,
            "mean": mean,
            "var": np.clip(var, 0.0, None),
        })
