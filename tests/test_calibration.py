"""Calibration: metrics, time grids, SQL time series and the error objectives (synthetic eplusout.sql, no EnergyPlus)."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from bdopt import Calibration, Channel, FieldSelector, Parameter, RangeParameter, split_time
from bdopt.calibration import cvrmse, nmbe
from bdopt.simulation.sql import RunResults


def make_sql(path, step_min=15, days=10, scale_el=1.0, offset_t=0.0, year=0):
    """Minimal eplusout.sql: a warm-up day, then ``days`` days; one electricity meter (J) and one zone temperature."""
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE ReportDataDictionary(ReportDataDictionaryIndex INTEGER, IsMeter INTEGER, Type TEXT, IndexGroup TEXT,
        TimestepType TEXT, KeyValue TEXT, Name TEXT, ReportingFrequency TEXT, ScheduleName TEXT, Units TEXT);
    CREATE TABLE Time(TimeIndex INTEGER, Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER,
        Dst INTEGER, Interval INTEGER, IntervalType INTEGER, SimulationDays INTEGER, DayType TEXT,
        EnvironmentPeriodIndex INTEGER, WarmupFlag INTEGER);
    CREATE TABLE ReportData(ReportDataIndex INTEGER, TimeIndex INTEGER, ReportDataDictionaryIndex INTEGER, Value REAL);
    CREATE TABLE EnvironmentPeriods(EnvironmentPeriodIndex INTEGER, SimulationIndex INTEGER, EnvironmentName TEXT, EnvironmentType INTEGER);
    """)
    con.execute("INSERT INTO EnvironmentPeriods VALUES (1,1,'DD',1),(2,1,'RUN',3)")
    f = "Hourly" if step_min == 60 else "Zone Timestep"
    con.execute("INSERT INTO ReportDataDictionary VALUES (1,1,'Sum','','Zone','','Electricity:Facility',?, '', 'J')", (f,))
    con.execute("INSERT INTO ReportDataDictionary VALUES (2,0,'Avg','','Zone','AULA','Zone Mean Air Temperature',?, '', 'C')", (f,))
    t0 = pd.Timestamp("2001-03-01")
    n = days * 24 * 60 // step_min
    ti = 0
    for k in range(-96, n):  # first day is warm-up
        stamp = t0 + pd.Timedelta(minutes=step_min * (k + 1))
        warm = 1 if k < 0 else 0
        ti += 1
        hour = stamp.hour + (24 if (stamp.hour == 0 and stamp.minute == 0) else 0)
        d = stamp - pd.Timedelta(days=1) if hour == 24 else stamp
        con.execute("INSERT INTO Time VALUES (?,?,?,?,?,?,0,?,1,1,'Mon',2,?)",
                    (ti, year, d.month, d.day, hour, stamp.minute, step_min, warm))
        kwh = (1.0 + np.sin(2 * np.pi * stamp.hour / 24) ** 2) * step_min / 60 * scale_el
        con.execute("INSERT INTO ReportData VALUES (?,?,1,?)", (2 * ti, ti, kwh * 3.6e6))
        con.execute("INSERT INTO ReportData VALUES (?,?,2,?)", (2 * ti + 1, ti, 20 + 3 * np.sin(2 * np.pi * stamp.hour / 24) + offset_t))
    con.commit()
    con.close()


def measured(step, days=10, **kw):
    idx = pd.date_range("2001-03-01", periods=days * 24 * 60 // step, freq=f"{step}min") + pd.Timedelta(minutes=step)
    kwh = (1.0 + np.sin(2 * np.pi * idx.hour / 24) ** 2) * step / 60
    temp = 20 + 3 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.Series(kwh, idx), pd.Series(temp, idx)


def test_metrics():
    m = np.array([10.0, 20.0, 30.0])
    assert nmbe(m, m) == 0 and cvrmse(m, m) == 0
    assert nmbe(m * 1.1, m) == pytest.approx(100 * 0.1 * 60 / (2 * 20))
    assert cvrmse(m + 2, m) == pytest.approx(100 * np.sqrt(3 * 4 / 2) / 20)


def test_timeseries_skips_warmup_and_design_days(tmp_path):
    make_sql(tmp_path / "a.sql", step_min=60, days=2)
    s = RunResults(tmp_path / "a.sql").timeseries("Electricity:Facility", True, None, "Hourly", year=2001)
    assert len(s) == 48
    assert s.index[0] == pd.Timestamp("2001-03-01 01:00") and s.index[-1] == pd.Timestamp("2001-03-03 00:00")
    assert RunResults(tmp_path / "a.sql").timeseries("Nope", True, None, "Hourly").empty


@pytest.mark.parametrize("step", [15, 60])
def test_perfect_model_has_zero_error_at_any_resolution(tmp_path, step):
    make_sql(tmp_path / "a.sql", step_min=step)
    el, t = measured(step)
    run = RunResults(tmp_path / "a.sql")
    for ch in (Channel.energy("el", el, "Electricity:Facility"), Channel.temperature("t", t, "AULA")):
        g = ch.grid(ch.simulated(run, 2001))
        assert len(g) == len(ch.measured)
        assert np.allclose(g["sim"], g["meas"])


def test_mixed_resolution_sums_energy_and_converts_power(tmp_path):
    make_sql(tmp_path / "h.sql", step_min=60)
    run = RunResults(tmp_path / "h.sql")
    el15, _ = measured(15)  # 15-min kWh meter vs hourly simulation: compared hourly, energy summed
    ch = Channel.energy("el", el15, "Electricity:Facility")
    assert ch.sim_frequency == "Timestep"
    el60, _ = measured(60)
    ch = Channel.energy("el", el60 * 1.0, "Electricity:Facility", measured_unit="kW")  # hourly kW == kWh
    assert np.allclose(ch.measured.values, el60.values)
    ch_daily = Channel.energy("el", el60, "Electricity:Facility", compare="1D")
    g = ch_daily.grid(ch_daily.simulated(run, 2001))
    assert len(g) == 10 and np.allclose(g["sim"], g["meas"])
    # 15-min measurements against an hourly simulation are brought to the hourly grid
    make_sql(tmp_path / "q.sql", step_min=15)
    q = RunResults(tmp_path / "q.sql")
    hourly_sim = q.timeseries("Electricity:Facility", True, None, "Zone Timestep", 2001) / 3.6e6
    g = Channel.energy("el", el15, "Electricity:Facility").grid(hourly_sim)
    assert len(g) == len(el15) and np.allclose(g["sim"], g["meas"])


def test_gaps_in_measurements_drop_cells_not_bias_them(tmp_path):
    make_sql(tmp_path / "a.sql", step_min=60)
    el, _ = measured(60)
    el = el.copy()
    el.iloc[30:36] = np.nan  # six missing hours on day 2
    ch = Channel.energy("el", el, "Electricity:Facility", compare="1D")
    g = ch.grid(ch.simulated(RunResults(tmp_path / "a.sql"), 2001))
    assert len(g) == 9 and np.allclose(g["sim"], g["meas"])


def test_timestamp_start_convention(tmp_path):
    make_sql(tmp_path / "a.sql", step_min=60)
    el, _ = measured(60)
    shifted = pd.Series(el.values, el.index - pd.Timedelta(hours=1))  # same data stamped at interval start
    ch = Channel.energy("el", shifted, "Electricity:Facility", timestamp="start")
    g = ch.grid(ch.simulated(RunResults(tmp_path / "a.sql"), 2001))
    assert np.allclose(g["sim"], g["meas"]) and len(g) == 240


def test_tz_aware_measurements_are_rejected():
    s = pd.Series([1.0, 2.0, 3.0], pd.date_range("2001-03-01", periods=3, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="timezone"):
        Channel.temperature("t", s, "AULA")


def test_objectives_are_in_units_of_tolerance_and_use_training_period_only(tmp_path):
    make_sql(tmp_path / "a.sql", step_min=60, scale_el=1.2, offset_t=0.5)
    run = RunResults(tmp_path / "a.sql")
    el, t = measured(60)
    chans = [Channel.energy("el", el, "Electricity:Facility"), Channel.temperature("t", t, "AULA")]
    train, valid = split_time(chans, 0.5)
    pars = [Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="L")]
    # Calibration touches no file until evaluator() / report() is called
    cal = Calibration(pars, chans, "model.idf", "w.epw", train=train, validation=valid)
    assert cal.year == 2001 and cal.problem.objective_names == ["err_energy", "err_temperature"]
    e_el, e_t = (o.func(run) for o in cal.problem.objectives)
    assert e_t == pytest.approx(0.5 / 1.0)  # rmse 0.5 degC, tolerance 1 degC
    half = len(el) // 2  # training = first half only
    exp = cvrmse(el.values[:half] * 1.2, el.values[:half]) / 30.0
    assert e_el == pytest.approx(exp, rel=0.02)
    assert Calibration(pars, chans, "m.idf", "w.epw", groups="single").problem.objective_names == ["err_total"]
    assert Calibration(pars, chans, "m.idf", "w.epw", groups="channel").problem.objective_names == ["err_el", "err_t"]
    outs = cal.problem.objectives[0].idf_outputs()
    assert ("Output:Meter", "Electricity:Facility", "Hourly") in outs


def test_fingerprint_changes_with_data():
    el, _ = measured(60)
    a = Channel.energy("el", el, "Electricity:Facility").fingerprint()
    assert a == Channel.energy("el", el.copy(), "Electricity:Facility").fingerprint()
    assert a != Channel.energy("el", el * 1.01, "Electricity:Facility").fingerprint()
