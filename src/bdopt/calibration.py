"""Calibrate the model of an *existing* building against measurements.

This is the same machinery as design optimization (parameters, EnergyPlus, surrogates, NSGA-II), but the
objectives are not consumptions to minimise: they are **errors between simulation and measurement**, so the
optimum is the parameter set that fits the real building best. Nothing about the data is fixed: meters can be
hourly or sub-hourly, there can be any number of meters and temperature sensors, and the measured period can
be anything (it only has to lie inside the simulated run period).

::

    meas = pd.read_csv("misure.csv", index_col=0, parse_dates=True)
    channels = [
        Channel.energy("elettrico", meas["kWh_el"], "Electricity:Facility"),
        Channel.energy("calore", meas["kW_th"], "DistrictHeating:Facility", measured_unit="kW"),
        Channel.temperature("T_aula", meas["T_aula"], zones="AULA"),
        Channel.temperature("T_uffici", meas["T_uff"], zones=["UFF_1", "UFF_2"]),
    ]
    train, valid = split_time(channels, 0.7)
    cal = Calibration(parameters, channels, "model.idf", "weather_2024.epw", train=train, validation=valid)
    ev = cal.evaluator(processes=4)                          # EnergyPlusEvaluator on cal.problem
    result = AdaptiveSampler(cal.problem, ev).run(...)       # then GPSurrogate + nsga2, as in design studies
    best = cal.report(front.closest_to_ideal())              # metrics on train AND validation, + curves

Every objective is expressed in units of its **tolerance** (ASHRAE Guideline 14 by default), so ``1.0`` means
"just inside the limit" and the objectives of different quantities (kWh, degC) are directly comparable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .eplus.aliases import meter_candidates
from .objectives import FunctionReader
from .problem import Problem
from .simulation.sql import _FREQ_LABEL, RunResults

#: kWh in one unit of an energy-per-interval measurement
_ENERGY_TO_KWH = {"kWh": 1.0, "Wh": 1e-3, "MWh": 1e3, "J": 1 / 3.6e6, "kJ": 1 / 3600, "MJ": 1 / 3.6, "GJ": 1e3 / 3.6}
#: kW in one unit of a mean-power measurement
_POWER_TO_KW = {"kW": 1.0, "W": 1e-3, "MW": 1e3}
METRICS = ("cvrmse", "nmbe", "rmse", "mae")


# -- metrics -----------------------------------------------------------------------------------------------
def nmbe(sim, meas) -> float:
    """Normalised mean bias error in % (ASHRAE 14 with p=1); positive = the simulation over-predicts."""
    s, m = np.asarray(sim, float), np.asarray(meas, float)
    return float(100 * np.sum(s - m) / ((len(m) - 1) * np.mean(m)))


def cvrmse(sim, meas) -> float:
    """Coefficient of variation of the RMSE in % (ASHRAE 14 with p=1)."""
    s, m = np.asarray(sim, float), np.asarray(meas, float)
    return float(100 * np.sqrt(np.sum((s - m) ** 2) / (len(m) - 1)) / np.mean(m))


def rmse(sim, meas) -> float:
    """Root mean squared error, in the unit of the data (degC for temperatures)."""
    s, m = np.asarray(sim, float), np.asarray(meas, float)
    return float(np.sqrt(np.mean((s - m) ** 2)))


def mae(sim, meas) -> float:
    """Mean absolute error, in the unit of the data."""
    s, m = np.asarray(sim, float), np.asarray(meas, float)
    return float(np.mean(np.abs(s - m)))


_METRIC_FUNCS = {"cvrmse": cvrmse, "nmbe": nmbe, "rmse": rmse, "mae": mae}


def _step_of(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("need at least two samples to infer the time step")
    return pd.Series(index).diff().dropna().median()


def _is_fine(step: pd.Timedelta) -> bool:
    return step <= pd.Timedelta(hours=1)


# -- channel -----------------------------------------------------------------------------------------------
@dataclass
class Channel:
    """One measured quantity paired with the simulation output that should reproduce it.

    Prefer :meth:`energy` and :meth:`temperature`. ``measured`` is a ``pandas.Series`` with a tz-naive
    ``DatetimeIndex`` (local standard time, as in the weather file) at any regular resolution; gaps (NaN) are
    allowed. ``timestamp`` says whether a sample is stamped at the ``'end'`` (default) or the ``'start'`` of its
    interval. Simulation and measurement are put on a common grid, as coarse as the coarser of the two
    (or ``compare``, e.g. ``'1h'``, ``'1D'``, ``'MS'``), summing energies and averaging the rest.
    """

    name: str
    measured: pd.Series
    kind: str  # 'meter' | 'variable'
    sim_name: str
    sim_keys: list[str] = field(default_factory=list)
    group: str = "variable"
    agg: str = "mean"  # how to combine samples on the common grid: 'sum' | 'mean'
    metric: str = "rmse"
    scale: float = 1.0  # multiply the simulated value (J -> kWh for meters)
    compare: str | None = None
    weight: float = 1.0
    tolerance: float | None = None
    timestamp: str = "end"
    min_coverage: float = 0.9  # a grid cell with less measured data than this fraction is discarded

    def __post_init__(self) -> None:
        if self.kind not in ("meter", "variable"):
            raise ValueError("kind must be 'meter' or 'variable'")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.agg not in ("sum", "mean"):
            raise ValueError("agg must be 'sum' or 'mean'")
        if self.timestamp not in ("start", "end"):
            raise ValueError("timestamp must be 'start' or 'end'")
        m = pd.Series(self.measured, dtype=float).dropna()
        if not isinstance(m.index, pd.DatetimeIndex):
            raise TypeError(f"channel {self.name!r}: measured needs a DatetimeIndex")
        if m.index.tz is not None:
            raise ValueError(
                f"channel {self.name!r}: measured index is timezone-aware; convert it to local standard time "
                "and drop the timezone (series.tz_convert(...).tz_localize(None))"
            )
        self.measured = m[~m.index.duplicated()].sort_index()
        self.native_step = _step_of(self.measured.index)
        if self.tolerance is None:
            fine = _is_fine(self._grid_step_nominal())
            self.tolerance = {
                "cvrmse": 30.0 if fine else 15.0,
                "nmbe": 10.0 if fine else 5.0,
                "rmse": 1.0,
                "mae": 1.0,
            }[self.metric]

    # constructors --------------------------------------------------------------------------------------
    @classmethod
    def energy(
        cls,
        name: str,
        measured: pd.Series,
        meter: str,
        measured_unit: str = "kWh",
        metric: str = "cvrmse",
        **kw,
    ) -> Channel:
        """An energy meter. ``measured_unit`` is an energy per interval (``'kWh'``, ``'MJ'``, ...) or a mean
        power (``'kW'``, ``'W'``, ``'MW'``, converted with the sampling interval). ``meter`` is an EnergyPlus
        meter such as ``'Electricity:Facility'`` or ``'DistrictHeating:Facility'``."""
        m = pd.Series(measured, dtype=float)
        if measured_unit in _ENERGY_TO_KWH:
            m = m * _ENERGY_TO_KWH[measured_unit]
        elif measured_unit in _POWER_TO_KW:
            step_h = _step_of(m.dropna().index) / pd.Timedelta(hours=1)
            m = m * _POWER_TO_KW[measured_unit] * step_h
        else:
            raise ValueError(f"measured_unit must be one of {sorted({*_ENERGY_TO_KWH, *_POWER_TO_KW})}")
        return cls(name, m, "meter", meter, [], "energy", "sum", metric, 1 / 3.6e6, **kw)

    @classmethod
    def temperature(
        cls,
        name: str,
        measured: pd.Series,
        zones: str | Sequence[str],
        variable: str = "Zone Mean Air Temperature",
        metric: str = "rmse",
        **kw,
    ) -> Channel:
        """A temperature sensor (degC). If it stands for several zones, pass them all: the simulated
        value is their mean."""
        keys = [zones] if isinstance(zones, str) else list(zones)
        return cls(name, pd.Series(measured, dtype=float), "variable", variable, keys, "temperature", "mean",
                   metric, 1.0, **kw)

    @classmethod
    def variable(
        cls,
        name: str,
        measured: pd.Series,
        key: str | Sequence[str],
        variable: str,
        agg: str = "mean",
        metric: str = "rmse",
        scale: float = 1.0,
        **kw,
    ) -> Channel:
        """Any other ``Output:Variable`` (relative humidity, CO2, supply temperature ...)."""
        keys = [key] if isinstance(key, str) else list(key)
        return cls(name, pd.Series(measured, dtype=float), "variable", variable, keys, "variable", agg, metric,
                   scale, **kw)

    # IDF outputs / simulation ------------------------------------------------------------------------------
    @property
    def sim_frequency(self) -> str:
        """Reporting frequency requested from EnergyPlus: ``Timestep`` for sub-hourly data, else ``Hourly``."""
        return "Timestep" if self.native_step < pd.Timedelta(hours=1) else "Hourly"

    def _grid_step_nominal(self) -> pd.Timedelta:
        return max(self.native_step, pd.Timedelta(hours=1)) if self.compare is None else max(
            self.native_step, _nominal(self.compare))

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        """``Output:*`` objects EnergyPlus must report for this channel."""
        if self.kind == "meter":
            steam = idf is None or bool(idf.by_class("DistrictHeating:Steam"))
            return [("Output:Meter", c, self.sim_frequency) for c in meter_candidates(self.sim_name, version, steam)]
        return [("Output:Variable", k, self.sim_name, self.sim_frequency) for k in self.sim_keys]

    def simulated(self, run: RunResults, year: int) -> pd.Series:
        """The simulated series of this channel (end-of-interval stamps, already in the measurement unit)."""
        label = _FREQ_LABEL.get(self.sim_frequency.lower(), self.sim_frequency)
        if self.kind == "meter":
            parts = [run.timeseries(c, True, None, label, year) for c in meter_candidates(self.sim_name)]
            parts = [p for p in parts if len(p)]
            if not parts:
                raise KeyError(f"meter {self.sim_name!r} not reported by EnergyPlus (channel {self.name!r})")
            s = pd.concat(parts, axis=1).sum(axis=1)
        else:
            parts = [run.timeseries(self.sim_name, False, k, label, year) for k in self.sim_keys]
            missing = [k for k, p in zip(self.sim_keys, parts, strict=True) if not len(p)]
            if missing:
                raise KeyError(f"variable {self.sim_name!r} not reported for {missing} (channel {self.name!r})")
            s = pd.concat(parts, axis=1).mean(axis=1) if len(parts) > 1 else parts[0]
        return s * self.scale

    def grid(self, sim: pd.Series) -> pd.DataFrame:
        """Simulation and measurement on the common grid, columns ``sim`` and ``meas`` indexed by the cell start."""
        sim_step = _step_of(sim.index)
        step = max(sim_step, self.native_step)
        offset = pd.tseries.frequencies.to_offset(self.compare) if self.compare else pd.tseries.frequencies.to_offset(step)
        if self.compare and _nominal(self.compare) < step:
            offset = pd.tseries.frequencies.to_offset(step)
        meas = self._cells(self.measured, offset, self.native_step, self.timestamp, check=True)
        simc = self._cells(sim, offset, sim_step, "end", check=False)
        df = pd.concat({"sim": simc, "meas": meas}, axis=1).dropna()
        return df

    def _cells(self, s: pd.Series, offset, native: pd.Timedelta, stamp: str, check: bool) -> pd.Series:
        start = s.copy()
        if stamp == "end":
            start.index = start.index - native
        r = start.resample(offset)
        out = r.sum(min_count=1) if self.agg == "sum" else r.mean()
        if check:
            span = pd.Series(out.index.map(lambda t: (t + offset) - t), index=out.index)
            expected = span / native
            out = out[(r.count() >= self.min_coverage * expected).reindex(out.index).fillna(False)]
        return out

    def fingerprint(self) -> str:
        """Hash of everything about this channel that changes its error (used in the result cache key)."""
        h = hashlib.sha256()
        h.update(repr((self.name, self.kind, self.sim_name, self.sim_keys, self.agg, self.metric, self.scale,
                       self.compare, self.tolerance, self.timestamp, self.min_coverage)).encode())
        h.update(pd.util.hash_pandas_object(self.measured).values.tobytes())
        return h.hexdigest()


def _nominal(freq: str) -> pd.Timedelta:
    """Length of a pandas offset, taking a calendar offset (month ...) as its longest sensible value."""
    off = pd.tseries.frequencies.to_offset(freq)
    try:
        return pd.Timedelta(off)
    except (ValueError, TypeError):
        t0 = pd.Timestamp("2001-01-01")
        return (t0 + off) - t0


def split_time(channels: Sequence[Channel], train_fraction: float = 0.7) -> tuple[tuple, tuple]:
    """Split the span covered by the channels into ``(train, validation)`` periods ``(start, end)``.

    The first ``train_fraction`` of the span (rounded to midnight) trains, the rest validates. If the data cover
    more than a season, split by hand instead so that both periods contain similar conditions
    (e.g. train on winter and summer, validate on spring and autumn)."""
    lo = max(c.measured.index[0] for c in channels)
    hi = min(c.measured.index[-1] for c in channels)
    cut = (lo + (hi - lo) * train_fraction).normalize()
    return (lo.normalize(), cut), (cut, hi)


# -- the calibration problem -------------------------------------------------------------------------------
def _window(df: pd.DataFrame, period) -> pd.DataFrame:
    if period is None:
        return df
    a, b = (pd.Timestamp(p) if p is not None else None for p in period)
    if a is not None:
        df = df[df.index >= a]
    if b is not None:
        df = df[df.index < b]
    return df


class _ErrorObjective(FunctionReader):
    """Objective = weighted mean of the normalised errors (error / tolerance) of some channels, on the training period."""

    def __init__(self, cal: Calibration, channels: list[Channel], name: str, collect: bool = False):
        super().__init__(self._compute, name, direction="min")
        self.cal, self.channels, self.collect = cal, channels, collect
        self.fingerprint = hashlib.sha256(
            ("|".join(c.fingerprint() for c in channels) + repr((cal.train, cal.year))).encode()
        ).hexdigest()

    def _compute(self, run: RunResults) -> float:
        if self.collect:  # keep the full curves for Calibration.report
            self.cal._stash = {c.name: c.grid(c.simulated(run, self.cal.year)) for c in self.channels}
            return 0.0
        total = wsum = 0.0
        for c in self.channels:
            g = _window(c.grid(c.simulated(run, self.cal.year)), self.cal.train)
            if len(g) < 3:
                raise ValueError(f"channel {c.name!r}: only {len(g)} comparable samples in the training period "
                                 "(is the measured period inside the simulated run period?)")
            total += c.weight * abs(_METRIC_FUNCS[c.metric](g["sim"], g["meas"])) / c.tolerance
            wsum += c.weight
        return total / wsum

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        for c in self.channels:
            out += [o for o in c.idf_outputs(version, idf) if o not in out]
        return out


class Calibration:
    """A calibration study: unknown building parameters + measured channels -> a :class:`~bdopt.Problem`.

    Parameters
    ----------
    parameters : list of :class:`~bdopt.Parameter`
        the uncertain properties of the building (infiltration, internal gains, insulation, set points ...).
    channels : list of :class:`Channel`
    building, epw
        the model of the building and the weather of the **measured period** (a typical-year file makes a
        comparison with a specific period meaningless). The IDF ``RunPeriod`` must cover the measurements and
        its ``Timestep`` should be at least as fine as the finest measurement.
    train, validation : ``(start, end)`` or ``None``
        the period the objectives are computed on, and the held-out period only used by :meth:`report`.
        ``None`` = all measured data (and no validation). See :func:`split_time`.
    groups : ``'kind'`` (default), ``'channel'`` or ``'single'``
        one objective per quantity type (``err_energy``, ``err_temperature``: the trade-off between fitting
        consumption and temperature is a Pareto front), one per channel, or one weighted total.
    year : int, optional
        the calendar year of the run when the simulation has none (default: the first measured year).
    """

    def __init__(
        self,
        parameters: list,
        channels: Sequence[Channel],
        building,
        epw,
        train: tuple | None = None,
        validation: tuple | None = None,
        groups: str = "kind",
        year: int | None = None,
    ):
        if not channels:
            raise ValueError("at least one channel is needed")
        names = [c.name for c in channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        if groups not in ("kind", "channel", "single"):
            raise ValueError("groups must be 'kind', 'channel' or 'single'")
        self.parameters, self.channels = list(parameters), list(channels)
        self.building, self.epw = building, epw
        self.train, self.validation = train, validation
        self.year = int(year if year is not None else min(c.measured.index[0] for c in channels).year)
        self.groups = groups
        if groups == "channel":
            objs = [_ErrorObjective(self, [c], f"err_{c.name}") for c in self.channels]
        elif groups == "single":
            objs = [_ErrorObjective(self, self.channels, "err_total")]
        else:
            kinds: dict[str, list[Channel]] = {}
            for c in self.channels:
                kinds.setdefault(c.group, []).append(c)
            objs = [_ErrorObjective(self, cs, f"err_{k}") for k, cs in kinds.items()]
        self.problem = Problem(self.parameters, objs)
        self._stash: dict[str, pd.DataFrame] = {}

    def evaluator(self, **kw):
        """An :class:`~bdopt.EnergyPlusEvaluator` on :attr:`problem` (``processes``, ``out_dir`` ... as usual)."""
        from .evaluator import EnergyPlusEvaluator

        return EnergyPlusEvaluator(self.problem, self.building, self.epw, **kw)

    def report(self, params, **evaluator_kw) -> CalibrationReport:
        """Simulate one parameter set and compare with the measurements, on training **and** validation data.

        ``params`` is a mapping / one-row ``DataFrame`` / ``Series`` with a value per parameter (for example
        ``front.closest_to_ideal()`` or ``front.parameters.iloc[i]``). Nothing is taken from the cache: the run
        is repeated so the full curves are available."""
        from .evaluator import EnergyPlusEvaluator

        if isinstance(params, pd.DataFrame):
            params = params.iloc[0]
        row = pd.DataFrame([dict(params)])[[p.name for p in self.parameters]]
        probe = Problem(self.parameters, [_ErrorObjective(self, self.channels, "_collect", collect=True)])
        evaluator_kw.setdefault("out_dir", "bdopt_calibration_runs")
        ev = EnergyPlusEvaluator(probe, self.building, self.epw, cache=False, on_error="raise", **evaluator_kw)
        self._stash = {}
        ev.evaluate(row)
        return self._make_report(row.iloc[0].to_dict())

    def _make_report(self, parameters: dict) -> CalibrationReport:
        periods = {"train": self.train, "validation": self.validation}
        if self.train is None and self.validation is None:
            periods = {"all": None}
        elif self.validation is None:
            periods = {"train": self.train, "rest": None}
        rows = []
        for c in self.channels:
            full = self._stash[c.name]
            for pname, per in periods.items():
                g = _window(full, per) if pname != "rest" else full.drop(_window(full, self.train).index)
                if len(g) < 3:
                    continue
                m = {k: f(g["sim"], g["meas"]) for k, f in _METRIC_FUNCS.items()}
                err = abs(m[c.metric]) / c.tolerance
                ok = err <= 1.0 and (c.group != "energy" or abs(m["nmbe"]) <= (10.0 if _is_fine(c._grid_step_nominal()) else 5.0))
                rows.append({"channel": c.name, "period": pname, "n": len(g), **m,
                             "error": err, "ok": bool(ok)})
        return CalibrationReport(pd.DataFrame(rows), dict(self._stash), parameters, self.train, self.validation)


@dataclass
class CalibrationReport:
    """Result of :meth:`Calibration.report`.

    ``metrics``: one row per channel and period with ``n``, ``nmbe`` (%), ``cvrmse`` (%), ``rmse`` and ``mae``
    (data unit), ``error`` (the primary metric in units of its tolerance) and ``ok`` (inside the tolerance; for
    energy meters also inside the ASHRAE 14 NMBE limit). ``series``: ``channel -> DataFrame[sim, meas]`` on the
    comparison grid. A model that is good on ``train`` and poor on ``validation`` is over-fitted."""

    metrics: pd.DataFrame
    series: dict[str, pd.DataFrame]
    parameters: dict
    train: tuple | None = None
    validation: tuple | None = None

    def __str__(self) -> str:
        return self.metrics.round(3).to_string(index=False)

    def plot(self, channel: str | None = None):
        """Simulated vs measured curves (one panel per channel) with the training / validation split marked."""
        import matplotlib.pyplot as plt

        names = [channel] if channel else list(self.series)
        fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), squeeze=False)
        for ax, n in zip(axes[:, 0], names, strict=True):
            df = self.series[n]
            ax.plot(df.index, df["meas"], lw=1, label="measured")
            ax.plot(df.index, df["sim"], lw=1, label="simulated")
            if self.validation is not None and self.validation[0] is not None:
                ax.axvline(pd.Timestamp(self.validation[0]), color="k", ls=":", lw=1)
            ax.set_ylabel(n)
            ax.legend(loc="upper right")
        fig.tight_layout()
        return fig
