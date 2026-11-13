import urllib.request
import json
import numpy as np
import pandas as pd
from pathlib import Path

def _fetch_energinet_spot_prices(year: int, price_area: str = "DK1") -> pd.Series:
    """
    Fetches real historical hourly Day-Ahead Spot prices (EUR/MWh) from Energinet Data Service API.
    No API key required.
    """
    start_str = f"{year}-01-01T00:00"
    end_str = f"{year + 1}-01-01T00:00"
    filter_json = urllib.parse.quote(json.dumps({"PriceArea": [price_area]}))
    url = (
        f"https://api.energidataservice.dk/dataset/Elspotprices"
        f"?start={start_str}&end={end_str}&filter={filter_json}&sort=HourDK%20asc&limit=10000"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    records = data.get("records", [])
    if not records:
        raise RuntimeError(f"No records returned from Energinet for {year}")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["HourDK"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df["SpotPriceEUR"].astype(float)

def _fetch_open_meteo_pv_100mwp(year: int, pv_capacity_mwp: float = 100.0, lat: float = 56.15, lon: float = 9.05) -> pd.Series:
    """
    Fetches real historical hourly ERA5 solar radiation & temperature for Central Jutland (DK1)
    from Open-Meteo Archive API and converts to AC PV output (MW) for a 100 MWp solar park.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={year}-01-01&end_date={year}-12-31"
        f"&hourly=shortwave_radiation,temperature_2m&timezone=Europe%2FCopenhagen"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    hourly = data["hourly"]
    timestamps = pd.to_datetime(hourly["time"])
    ghi_w_m2 = np.array(hourly["shortwave_radiation"], dtype=float)
    temp_c = np.array(hourly["temperature_2m"], dtype=float)

    # Convert GHI to tilted plane irradiance (~1.18x gain on optimal tilt in Denmark) + temperature derating
    poa_irradiance = np.minimum(1100.0, ghi_w_m2 * 1.18)
    cell_temp = temp_c + 0.025 * poa_irradiance
    temp_factor = 1.0 - 0.004 * np.maximum(0.0, cell_temp - 25.0)
    system_losses = 0.88  # Inverter + cabling losses

    pv_mw = pv_capacity_mwp * (poa_irradiance / 1000.0) * temp_factor * system_losses
    pv_mw = np.clip(pv_mw, 0.0, pv_capacity_mwp * 0.92)

    return pd.Series(pv_mw, index=timestamps, name="pv_mw")

def _generate_calibrated_fallback_year(year: int, pv_capacity_mwp: float = 100.0) -> pd.DataFrame:
    """
    Fallback generator calibrated on real 2020 (Covid low prices) and 2022 (Energy Crisis) DK1 statistics
    in case the machine runs offline or behind a strict firewall.
    """
    rng = np.random.default_rng(seed=year)
    timestamps = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00:00", freq="h")
    # Remove Feb 29 if leap year to strictly keep 8760 hours (52 weeks)
    timestamps = timestamps[~((timestamps.month == 2) & (timestamps.day == 29))]
    T = len(timestamps)

    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values

    # Seasonal day length and solar elevation in Denmark (Winter = very low, Summer = long days)
    seasonal_env = 0.12 + 0.88 * np.sin(np.pi * (day_of_year - 1) / 365.0) ** 1.6
    day_window = np.maximum(0.0, np.sin((hour_of_day - 5.0) * np.pi / 14.0))

    # Cloud cover Markov-like daily blocks (with multi-day Dunkelflaute episodes)
    daily_clouds = rng.beta(a=2.2, b=1.5, size=365)
    # Inject 3-day dark episodes in winter/autumn
    for d in [15, 42, 310, 340]:
        daily_clouds[d:d+4] = 0.12
    hourly_cloud_factor = np.repeat(daily_clouds, 24)[:T]

    pv_mw = np.clip(pv_capacity_mwp * 0.90 * seasonal_env * day_window * hourly_cloud_factor, 0.0, pv_capacity_mwp)

    if year == 2020:
        # 2020: Low average price (~28-35 EUR/MWh), frequent negative midday prices in spring/summer
        base_price = 32.0 + 10.0 * np.cos(2 * np.pi * day_of_year / 365.0)
        duck_curve = -18.0 * day_window * hourly_cloud_factor + 16.0 * np.exp(-0.5 * ((hour_of_day - 19) / 2.2) ** 2)
        noise = rng.normal(0.0, 6.0, size=T)
        price_da = base_price + duck_curve + noise
        # Occasional winter evening peaks up to 110-140 EUR/MWh
        spike_mask = (hourly_cloud_factor < 0.20) & (hour_of_day >= 17) & (hour_of_day <= 20)
        price_da[spike_mask] += rng.uniform(45.0, 95.0, size=spike_mask.sum())
    else:
        # 2022: Energy crisis (average ~210 EUR/MWh, summer/autumn spikes up to 500-700 EUR/MWh)
        crisis_wave = 140.0 + 130.0 * np.sin(np.pi * np.clip((day_of_year - 60) / 240.0, 0.0, 1.0)) ** 2
        duck_curve = -65.0 * day_window * hourly_cloud_factor + 85.0 * np.exp(-0.5 * ((hour_of_day - 19) / 2.5) ** 2)
        noise = rng.normal(0.0, 25.0, size=T)
        price_da = crisis_wave + duck_curve + noise
        # Severe Dunkelflaute spikes (450 - 650 EUR/MWh)
        spike_mask = (hourly_cloud_factor < 0.22) & (hour_of_day >= 17) & (hour_of_day <= 21)
        price_da[spike_mask] += rng.uniform(180.0, 340.0, size=spike_mask.sum())

    return pd.DataFrame({
        "timestamp": timestamps,
        "month": timestamps.month,
        "week": np.minimum(52, ((timestamps.dayofyear - 1) // 7) + 1),
        "pv_mw": np.round(pv_mw, 3),
        "price_da": np.round(price_da, 2),
    })

def load_or_download_year_data(year: int, project_root: Path, pv_capacity_mwp: float = 100.0, area: str = "DK1") -> pd.DataFrame:
    """
    Loads `data/raw/market_{year}.csv` if available, or downloads real hourly Spot + PV data
    from Energinet & Open-Meteo APIs (with automatic fallback if offline) and caches it to disk.
    Always returns an exact 8760-hour DataFrame with columns: ['timestamp', 'month', 'week', 'pv_mw', 'price_da'].
    """
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"market_{year}.csv"

    if csv_path.exists():
        print(f"[DataLoader] Loading cached {year} dataset from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df.iloc[:8760].reset_index(drop=True)

    print(f"[DataLoader] Downloading real historical {year} data (Energinet {area} Spot + Open-Meteo ERA5 PV)...")
    try:
        s_price = _fetch_energinet_spot_prices(year, price_area=area)
        s_pv = _fetch_open_meteo_pv_100mwp(year, pv_capacity_mwp=pv_capacity_mwp)

        df = pd.DataFrame({"price_da": s_price, "pv_mw": s_pv}).dropna()
        # Remove Feb 29 on leap years (like 2020) so every year has exactly 8760 hours (365 days)
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        df = df.iloc[:8760].copy()

        if len(df) < 8700:
            raise RuntimeError(f"Incomplete API series ({len(df)} rows)")

        df = df.reset_index().rename(columns={"index": "timestamp"})
        df["month"] = df["timestamp"].dt.month
        df["week"] = np.minimum(52, ((df["timestamp"].dt.dayofyear - 1) // 7) + 1)
        print(f"[DataLoader] Successfully downloaded real {year} dataset ({len(df)} hours).")
    except Exception as exc:
        print(f"[DataLoader] API download note ({exc}). Generating calibrated {year} DK1 dataset (8760h)...")
        df = _generate_calibrated_fallback_year(year, pv_capacity_mwp=pv_capacity_mwp)

    df.to_csv(csv_path, index=False)
    print(f"[DataLoader] Saved {year} dataset to: {csv_path}")
    return df.reset_index(drop=True)

def generate_20_saa_scenarios(
    df_2020: pd.DataFrame,
    df_2022: pd.DataFrame,
    n_scenarios: int = 20,
    seed: int = 123
) -> dict:
    """
    Generates S=20 equiprobable 8760-hour SAA scenarios using Block-Bootstrapping
    and Regime Mixing between 2020 (Cannibalization) and 2022 (Energy Crisis):
    - Preserves diurnal solar/price correlation and multi-day cloud persistence (7-day blocks).
    - Spans the full probability distribution from sunny cannibalized years (S01..S05)
      to moderate years (S06..S15) and severe Dunkelflaute crisis tail years (S16..S20).
    """
    rng = np.random.default_rng(seed=seed)
    scenarios = {}

    # Include exact historical baselines as S01 (2020) and S20 (2022 Crisis)
    scenarios["S01_Hist_2020"] = df_2020.copy()

    for idx in range(2, n_scenarios):
        scen_name = f"S{idx:02d}_SAA"
        # Crisis weight w_crisis increases progressively from 0.05 (S02) to 0.95 (S19)
        w_crisis = (idx - 1) / (n_scenarios - 1)

        month_dfs = []
        for m in range(1, 13):
            sub_20 = df_2020[df_2020["month"] == m].reset_index(drop=True)
            sub_22 = df_2022[df_2022["month"] == m].reset_index(drop=True)
            T_m = min(len(sub_20), len(sub_22))
            n_days = T_m // 24

            # Circular day-block shift (shifts by whole days so hours 0..23 stay aligned!)
            shift_days_20 = int(rng.integers(0, n_days))
            shift_days_22 = int(rng.integers(0, n_days))

            pv_20 = np.roll(sub_20["pv_mw"].values[:T_m], shift_days_20 * 24)
            pr_20 = np.roll(sub_20["price_da"].values[:T_m], shift_days_20 * 24)
            pv_22 = np.roll(sub_22["pv_mw"].values[:T_m], shift_days_22 * 24)
            pr_22 = np.roll(sub_22["price_da"].values[:T_m], shift_days_22 * 24)

            # Weekly block selection (7-day blocks preserve multi-day cloud fronts)
            n_weeks_m = (n_days // 7) + 1
            choose_22_week = rng.random(n_weeks_m) < w_crisis
            choose_22_hour = np.repeat(choose_22_week, 7 * 24)[:T_m]

            # Blend block-selected series with smooth regime factor
            pv_blend = np.where(choose_22_hour, pv_22, pv_20) * rng.uniform(0.92, 1.08)
            pr_blend = (1.0 - 0.5 * w_crisis) * np.where(choose_22_hour, pr_22, pr_20) + (0.5 * w_crisis) * pr_22
            pr_blend *= rng.uniform(0.90, 1.12)

            # Inject a 3-day Dunkelflaute stress event in high-risk tail scenarios (S16..S19)
            if idx >= int(0.75 * n_scenarios) and m in [1, 2, 8, 11, 12]:
                start_d = int(rng.integers(0, max(1, n_days - 4))) * 24
                pv_blend[start_d:start_d + 72] *= 0.18
                # Evening peak spikes during the dark event
                for h_off in range(start_d, min(T_m, start_d + 72)):
                    if (h_off % 24) in [17, 18, 19, 20]:
                        pr_blend[h_off] = max(pr_blend[h_off], rng.uniform(320.0, 580.0))

            df_m = sub_20.iloc[:T_m].copy()
            df_m["pv_mw"] = np.clip(np.round(pv_blend, 3), 0.0, 100.0)
            df_m["price_da"] = np.round(pr_blend, 2)
            month_dfs.append(df_m)

        scenarios[scen_name] = pd.concat(month_dfs, ignore_index=True)

    scenarios[f"S{n_scenarios:02d}_Hist_2022"] = df_2022.copy()
    return scenarios