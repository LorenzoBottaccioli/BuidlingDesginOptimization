"""Read results from ``eplusout.sql``.

The ``ReportDataDictionary`` / ``ReportData`` tables have the same layout in every EnergyPlus
version from 8.7 to 25.x (checked on 8.7, 9.0, 9.6, 23.1, 24.2 and 25.2), so no per-version branch
is needed here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..objectives import ENERGY_UNITS, FunctionReader, MeterReader, Objective, VariableReader

_AGG = {"sum": np.sum, "max": np.max, "mean": np.mean}
_FREQ_LABEL = {
    "runperiod": "Run Period",
    "annual": "Annual",
    "monthly": "Monthly",
    "daily": "Daily",
    "hourly": "Hourly",
    "timestep": "Zone Timestep",
    "detailed": "HVAC System Timestep",
}


class MissingOutput(KeyError):
    """A requested meter or variable was not reported (and ``missing="error"``)."""
    pass


@dataclass
class RunResults:
    """Query interface over one run's ``eplusout.sql``."""

    sql_path: Path

    def _connect(self) -> sqlite3.Connection:
        if not Path(self.sql_path).is_file():
            raise FileNotFoundError(f"{self.sql_path} not found (did EnergyPlus fail?)")
        return sqlite3.connect(f"file:{self.sql_path}?mode=ro", uri=True)

    def series(
        self, name: str, is_meter: bool, key: str | None = None, frequency: str = "Run Period"
    ) -> tuple[np.ndarray, str | None]:
        """Values and units of a meter/variable at a reporting frequency (empty array if absent)."""
        q = (
            "SELECT d.ReportDataDictionaryIndex, d.Units FROM ReportDataDictionary d "
            "WHERE lower(d.Name)=lower(?) AND d.IsMeter=? AND d.ReportingFrequency=?"
        )
        args: list = [name, 1 if is_meter else 0, frequency]
        if key not in (None, "*"):
            q += " AND lower(d.KeyValue)=lower(?)"
            args.append(key)
        with self._connect() as con:
            rows = con.execute(q, args).fetchall()
            if not rows:
                return np.array([]), None
            ids = [r[0] for r in rows]
            marks = ",".join("?" * len(ids))
            vals = con.execute(
                f"SELECT Value FROM ReportData WHERE ReportDataDictionaryIndex IN ({marks})", ids
            ).fetchall()
        return np.array([v[0] for v in vals], dtype=float), rows[0][1]

    def meter(self, name: str, frequency: str = "Run Period") -> np.ndarray:
        """Values of a meter at a reporting frequency (default: run period)."""
        return self.series(name, True, None, frequency)[0]

    def variable(self, key: str, name: str, frequency: str = "Run Period") -> np.ndarray:
        """Values of an output variable for a key at a reporting frequency."""
        return self.series(name, False, key, frequency)[0]

    def timeseries(
        self, name: str, is_meter: bool, key: str | None = None, frequency: str = "Hourly", year: int = 2001
    ) -> pd.Series:
        """Time-stamped values of a meter/variable (empty series if absent).

        The index is a ``DatetimeIndex`` of the *end* of each interval (EnergyPlus convention: ``Hour=24`` is
        midnight of the next day). Only the weather-file run period is read: warm-up days and design days
        are left out. ``year`` replaces the year when the run has none (``0``, i.e. a typical-year file).
        If several keys match (``key=None`` or ``'*'``) their values are summed per timestamp.
        """
        q = (
            "SELECT d.ReportDataDictionaryIndex FROM ReportDataDictionary d "
            "WHERE lower(d.Name)=lower(?) AND d.IsMeter=? AND d.ReportingFrequency=?"
        )
        args: list = [name, 1 if is_meter else 0, frequency]
        if key not in (None, "*"):
            q += " AND lower(d.KeyValue)=lower(?)"
            args.append(key)
        with self._connect() as con:
            ids = [r[0] for r in con.execute(q, args).fetchall()]
            if not ids:
                return pd.Series(dtype=float)
            marks = ",".join("?" * len(ids))
            envs = [r[0] for r in con.execute(
                "SELECT EnvironmentPeriodIndex FROM EnvironmentPeriods WHERE EnvironmentType=3").fetchall()]
            where = f"r.ReportDataDictionaryIndex IN ({marks}) AND (t.WarmupFlag IS NULL OR t.WarmupFlag=0)"
            if envs:
                where += f" AND t.EnvironmentPeriodIndex IN ({','.join('?' * len(envs))})"
            rows = con.execute(
                "SELECT t.Year, t.Month, t.Day, t.Hour, t.Minute, r.Value FROM ReportData r "
                f"JOIN Time t ON r.TimeIndex=t.TimeIndex WHERE {where} ORDER BY t.TimeIndex",
                [*ids, *envs],
            ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        a = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
        yr = np.where(np.nan_to_num(a[:, 0]) > 0, a[:, 0], year).astype(int)
        mon = np.where(np.nan_to_num(a[:, 1]) > 0, a[:, 1], 1).astype(int)
        day = np.where(np.nan_to_num(a[:, 2]) > 0, a[:, 2], 1).astype(int)
        base = pd.to_datetime({"year": yr, "month": mon, "day": day})
        idx = base + pd.to_timedelta(np.nan_to_num(a[:, 3]), unit="h") + pd.to_timedelta(np.nan_to_num(a[:, 4]), unit="min")
        return pd.Series(a[:, 5], index=pd.DatetimeIndex(idx)).groupby(level=0).sum().sort_index()


def read_objective(obj: Objective, run: RunResults, warnings: list[str]) -> float:
    """Value of one objective for one run."""
    if isinstance(obj, FunctionReader):
        return float(obj.func(run))
    if isinstance(obj, MeterReader):
        total, found = 0.0, False
        for cand in obj.candidates:
            vals, units = run.series(cand, True, None, "Run Period")
            if vals.size == 0:
                continue
            found = True
            v = float(_AGG[obj.aggregate](vals))
            if units == "J":
                v *= ENERGY_UNITS[obj.unit]
            total += v
        if found:
            return total
        msg = (
            f"meter {obj.meter!r} not reported by EnergyPlus (no consumption, or unknown name)"
        )
        if obj.missing == "error":
            raise MissingOutput(msg)
        warnings.append(msg)
        return 0.0 if obj.missing == "zero" else float("nan")
    if isinstance(obj, VariableReader):
        label = _FREQ_LABEL.get(obj.frequency.lower(), obj.frequency)
        vals, _ = run.series(obj.variable, False, obj.key_value, label)
        if vals.size:
            return float(_AGG[obj.aggregate](vals))
        msg = f"variable {obj.variable!r} (key {obj.key_value!r}) not reported by EnergyPlus"
        if obj.missing == "error":
            raise MissingOutput(msg)
        warnings.append(msg)
        return 0.0 if obj.missing == "zero" else float("nan")
    raise TypeError(f"unsupported objective {obj!r}")
