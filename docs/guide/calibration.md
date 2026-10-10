# Calibrating the model of an existing building

Design optimization asks *"which design consumes least?"*. Calibration asks *"which parameters make the model
look most like this real building?"*. The machinery is the same (parameters, EnergyPlus, adaptive sampling,
surrogates, NSGA-II); the objectives are **errors between simulation and measurement**, not consumptions.

The data are not standardised, and nothing here assumes they are: meters may be hourly or sub-hourly, there is
any number of meters and temperature sensors, and the measured period is whatever you have (it must lie inside
the run period of the model).

```python
import pandas as pd
from bdopt import *

meas = pd.read_csv("misure.csv", index_col=0, parse_dates=True)       # tz-naive, local standard time
channels = [
    Channel.energy("elettrico", meas["kWh_el"], "Electricity:Facility"),
    Channel.energy("calore", meas["kW_th"], "DistrictHeating:Facility", measured_unit="kW"),  # mean power
    Channel.temperature("T_aula", meas["T_aula"], zones="AULA"),
    Channel.temperature("T_uffici", meas["T_uff"], zones=["UFF_1", "UFF_2"]),                 # mean of the zones
]
train, valid = split_time(channels, 0.7)
cal = Calibration(parameters, channels, "model.idf", "weather_of_the_measured_year.epw",
                  train=train, validation=valid)            # groups="kind": err_energy and err_temperature

ev = cal.evaluator(processes=4)
res = AdaptiveSampler(cal.problem, ev).run(n_initial=30, n_test=8, max_iter=40)
front = nsga2(cal.problem, GPSurrogate().fit(res.data, cal.problem))
rep = cal.report(front.closest_to_ideal())       # re-simulates ONE design with EnergyPlus
print(rep)                                                  # metrics on train and on validation
rep.plot()                                                  # measured vs simulated curves
```

## What is compared

* Simulation and measurement are put on a common time grid, as coarse as the coarser of the two (or `compare=`
  `"1h"`, `"1D"`, `"MS"` ...). Energy is **summed**, temperatures are **averaged**. A 15-minute meter against an
  hourly simulation is compared hourly. A grid cell with less than 90 % of the measured data present is dropped,
  so gaps do not bias a sum.
* `Channel.energy(..., measured_unit=)` accepts energy per interval (`kWh`, `MWh`, `MJ` ...) or mean power
  (`kW`, `W`, `MW`, converted with the sampling interval). `timestamp="start"` for meters stamped at the start of
  the interval (default: end, as EnergyPlus).
* For sub-hourly data the channel asks EnergyPlus for `Timestep` reporting; set the IDF `Timestep` at least as
  fine as the finest measurement. The IDF `RunPeriod` must cover the measurements and the weather file must be
  the one of the measured period.

## Objectives and tolerances

Each objective is a weighted mean of `|error| / tolerance`, so **1.0 = just inside the limit** and kWh and °C
are comparable. Defaults follow ASHRAE Guideline 14: CV(RMSE) 30 % / NMBE 10 % for hourly or finer comparison,
15 % / 5 % for monthly or coarser; 1 °C RMSE for temperatures. Override per channel (`tolerance=`, `metric=`,
`weight=`).

`groups="kind"` (default) gives one objective per quantity type, so fitting consumption versus fitting
temperature becomes a **Pareto front**; if the two cannot both be satisfied the model is usually structurally
wrong (schedules, missing system), not badly parameterised. `groups="channel"` gives one objective per
channel, `groups="single"` one weighted total.

## Validation

Objectives use the training period only. `report()` evaluates one design on training **and** validation data;
a good training fit with a poor validation fit means over-fitting. Prefer periods that contain similar
conditions (e.g. train on winter + summer, validate on spring + autumn) rather than a plain chronological split
when you have more than one season. A year of monthly meters alone cannot identify many parameters: screen
for the 6-10 that matter first (a Latin hypercube is enough).

Calibration results are cached like any other study; the key includes the measured data and the training period.
