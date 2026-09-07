# NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)

This directory stores the official NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset used by **TurboLife AI**.

---

## Dataset Overview

The dataset simulates the operational life and degradation trajectories of commercial turbofan jet engines under sea-level conditions with high pressure compressor (HPC) wear.

| File | Description | Trajectories | Format |
| :--- | :--- | :--- | :--- |
| `train_FD001.txt` | Complete run-to-failure engine trajectories | 100 Engines (20,631 cycles) | Space-delimited |
| `test_FD001.txt` | Partial operational trajectories halted prior to failure | 100 Engines (13,096 cycles) | Space-delimited |
| `RUL_FD001.txt` | Ground-truth Remaining Useful Life (RUL) for each test engine | 100 True RUL values | Single column |

---

## Data Schema (26 Columns)

The raw text files contain 26 space-separated columns without header rows:

1. **`engine_id`**: Numerical identifier of the aircraft engine unit (1 to 100).
2. **`cycle`**: Current flight/mission operating cycle (time step $t$).
3. **`setting_1`**: Operational Setting 1 (Altitude / Mach Number).
4. **`setting_2`**: Operational Setting 2 (Throttle Resolver Angle).
5. **`setting_3`**: Operational Setting 3 (Sea-level condition constant).
6. **`sensor_1` – `sensor_21`**: 21 thermodynamic and mechanical sensor measurements.

---

## Turbofan Sensor Description Table

| Sensor ID | Sensor Symbol | Engineering Description | Units | Informative in FD001? |
| :--- | :--- | :--- | :--- | :--- |
| `sensor_1` | T2 | Total Temperature at Fan Inlet | °R | ❌ Static (Constant) |
| `sensor_2` | T24 | Total Temperature at LPC Outlet | °R | ✅ **Informative** |
| `sensor_3` | T30 | Total Temperature at HPC Outlet | °R | ✅ **Informative** |
| `sensor_4` | T50 | Total Temperature at LPT Outlet | °R | ✅ **Informative** |
| `sensor_5` | P2 | Total Pressure at Fan Inlet | psia | ❌ Static |
| `sensor_6` | P15 | Total Pressure in Bypass Duct | psia | ❌ Static |
| `sensor_7` | P30 | Total Pressure at HPC Outlet | psia | ✅ **Informative** |
| `sensor_8` | Nf | Physical Fan Speed | rpm | ✅ **Informative** |
| `sensor_9` | Nc | Physical Core Speed | rpm | ✅ **Informative** |
| `sensor_10` | epr | Engine Pressure Ratio ($P_{50} / P_2$) | - | ❌ Static |
| `sensor_11` | Ps30 | Static Pressure at HPC Outlet | psia | ✅ **Informative** |
| `sensor_12` | phi | Ratio of Fuel Flow to $P_{s30}$ | pps/psia | ✅ **Informative** |
| `sensor_13` | NRf | Corrected Fan Speed | rpm | ✅ **Informative** |
| `sensor_14` | NRc | Corrected Core Speed | rpm | ✅ **Informative** |
| `sensor_15` | BPR | Bypass Ratio | - | ✅ **Informative** |
| `sensor_16` | farB | Burner Fuel-Air Ratio | - | ❌ Static |
| `sensor_17` | htBleed | Bleed Enthalpy | - | ✅ **Informative** |
| `sensor_18` | Nf_dmd | Demanded Fan Speed | rpm | ❌ Static |
| `sensor_19` | PCNfR_dmd | Demanded Corrected Fan Speed | rpm | ❌ Static |
| `sensor_20` | W31 | HPT Coolant Bleed | lbm/s | ✅ **Informative** |
| `sensor_21` | W32 | LPT Coolant Bleed | lbm/s | ✅ **Informative** |

---

## Manual Download Instructions

If automatic download is unavailable in your environment:
1. Download the full C-MAPSS dataset from the NASA repository or Kaggle:
   - NASA Archive: [Turbofan Engine Degradation Simulation Data Set](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)
2. Extract the archive.
3. Copy `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` directly into this `data/` directory.
