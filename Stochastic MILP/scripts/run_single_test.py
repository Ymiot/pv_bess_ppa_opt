import sys
import yaml
from pathlib import Path

# Project root setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

config_path = project_root / "data" / "config" / "project_params.yml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

import numpy as np
import pandas as pd
from src.stochastic_ems import solve_two_stage_stochastic_ems
from src.finance import evaluate_project_finance
from src.plotting import (
    plot_hourly_dispatch,
    plot_contract_profiles_comparison,
    plot_risk_metrics_comparison,
    plot_bess_cycling_and_utilization,
    plot_penalties_and_allowances_analysis,
)

def generate_five_scenarios(T: int = 168) -> dict:
    """
    Generates 5 realistic weather/price scenarios (168h = 1 week) so P50, P90, VaR90, and CVaR90
    are well-differentiated across the probability distribution.
    """
    np.random.seed(42)
    hours = np.arange(T)
    base_pv = np.maximum(0, 85.0 * np.sin((hours % 24 - 6) * np.pi / 12))

    scenarios = {}

    # 1. Sunny Cannibalization (High PV, negative midday prices, moderate evening peak)
    p_cannib = np.maximum(-15.0, 40 - 55 * np.sin((hours % 24 - 8) * np.pi / 12) + np.random.normal(0, 4, T))
    scenarios["S1_Sunny_Cannib"] = pd.DataFrame({"pv_mw": base_pv * 1.05, "price_da": p_cannib})

    # 2. Calm 2020 Baseline (Normal solar, moderate Spot prices)
    p_2020 = 50 - 35 * np.sin((hours % 24 - 8) * np.pi / 12) + np.random.normal(0, 5, T)
    scenarios["S2_Calm_2020"] = pd.DataFrame({"pv_mw": base_pv * 0.90, "price_da": p_2020})

    # 3. Variable Spring/Autumn (Intermittent clouds, strong evening spread for BESS cycling)
    pv_var = base_pv * np.where((hours // 24) % 2 == 0, 0.95, 0.45)
    p_var = 75 - 50 * np.sin((hours % 24 - 8) * np.pi / 12) + np.random.normal(0, 8, T)
    scenarios["S3_Variable_Spread"] = pd.DataFrame({"pv_mw": pv_var, "price_da": p_var})

    # 4. Cloudy Winter Week (Low solar, elevated Spot prices around 110-180 €/MWh)
    p_cloudy = 95 - 40 * np.sin((hours % 24 - 8) * np.pi / 12) + np.random.normal(0, 10, T)
    scenarios["S4_Cloudy_Winter"] = pd.DataFrame({"pv_mw": base_pv * 0.50, "price_da": p_cloudy})

    # 5. Crisis 2022 Dunkelflaute (3 dark days + extreme Spot spikes at 450 and 520 €/MWh)
    pv_crisis = base_pv.copy()
    pv_crisis[48:120] *= 0.15
    p_2022 = 115 - 55 * np.sin((hours % 24 - 8) * np.pi / 12)
    p_2022[65:70] = 450.0
    p_2022[89:94] = 520.0
    scenarios["S5_Crisis_2022"] = pd.DataFrame({"pv_mw": pv_crisis, "price_da": p_2022})

    return scenarios

def main():
    scenarios = generate_five_scenarios(T=168)
    test_run_name = "run_single_test"

    # Toggle BESS sizing optimization here (True = co-optimize MW/MWh, False = fixed 50MW/100MWh)
    OPTIMIZE_BESS = True

    summary_rows = []
    all_ems_results = {}

    # -------------------------------------------------------------------------
    # STEP 1: Run the Direct Market Sale Benchmark ("Merchant_Only", 0 Baseload)
    # -------------------------------------------------------------------------
    print("\n================ BENCHMARK: MERCHANT_ONLY (DIRECT SPOT SALE) ================")
    res_merch = solve_two_stage_stochastic_ems(
        scenarios_data=scenarios,
        cfg=cfg,
        allowance_pct=0.0,
        contract_profile="Merchant_Only",
        fixed_baseload_mw=0.0,
        optimize_bess_size=OPTIMIZE_BESS,
        solver_name="appsi_highs"
    )
    fin_merch = evaluate_project_finance(
        bankable_ppa_revenue_eur=0.0,
        merchant_bonus_revenue_eur=res_merch["Annualized_P90_Merchant_MEUR"] * 1e6,
        pv_mw=cfg["project"]["pv_capacity_mwp"],
        bess_power_mw=res_merch["Optimal_BESS_Power_MW"],
        bess_energy_mwh=res_merch["Optimal_BESS_Energy_MWh"],
        cfg=cfg
    )
    all_ems_results[("Merchant_Only", 0.0)] = res_merch
    row_m = {k: v for k, v in {**res_merch, **fin_merch}.items() if not isinstance(v, dict)}
    summary_rows.append(row_m)

    print(
        f"Merchant_Only | BESS: {res_merch['Optimal_BESS_Power_MW']:.1f}MW/{res_merch['Optimal_BESS_Energy_MWh']:.1f}MWh | "
        f"Cycles: {res_merch['Annual_BESS_Cycles']:.0f}/yr | "
        f"Spot Rev P50/P90/CVaR90: {res_merch['Annualized_P50_Total_MEUR']:.2f} / {res_merch['Annualized_P90_Total_MEUR']:.2f} / {res_merch['Annualized_CVaR90_Total_MEUR']:.2f} M€ | "
        f"IRR Total (P90): {fin_merch['IRR_Total_pct']:.2f}%"
    )
    plot_hourly_dispatch(
        res_merch["Hourly_Results"]["S5_Crisis_2022"],
        "S5_Crisis_2022", test_run_name, "Merchant_Only", 0.0, res_merch["Annual_BESS_Cycles"]
    )

    # -------------------------------------------------------------------------
    # STEP 2: Run the 4 PPA Contract Profiles across Allowance Quotas
    # -------------------------------------------------------------------------
    profiles_to_test = ["24/7", "08h-20h", "12h-00h", "18h-20h"]
    allowance_pcts = [0.00, 0.05, 0.10, 0.15]

    for profile in profiles_to_test:
        print(f"\n================ CONTRACT PROFILE: {profile} (Fixed PPA = 75 €/MWh) ================")
        for allow_pct in allowance_pcts:
            res_ems = solve_two_stage_stochastic_ems(
                scenarios_data=scenarios,
                cfg=cfg,
                allowance_pct=allow_pct,
                contract_profile=profile,
                fixed_baseload_mw=None,
                optimize_bess_size=OPTIMIZE_BESS,
                solver_name="appsi_highs"
            )

            fin = evaluate_project_finance(
                bankable_ppa_revenue_eur=res_ems["Annualized_P90_Bankable_MEUR"] * 1e6,
                merchant_bonus_revenue_eur=res_ems["Annualized_P90_Merchant_MEUR"] * 1e6,
                pv_mw=cfg["project"]["pv_capacity_mwp"],
                bess_power_mw=res_ems["Optimal_BESS_Power_MW"],
                bess_energy_mwh=res_ems["Optimal_BESS_Energy_MWh"],
                cfg=cfg
            )

            all_ems_results[(profile, allow_pct)] = res_ems
            row = {k: v for k, v in {**res_ems, **fin}.items() if not isinstance(v, dict)}
            summary_rows.append(row)

            print(
                f"Allow: {allow_pct*100:3.0f}% ({res_ems['Allowance_Hours']:2d}h) | "
                f"BL: {res_ems['Optimal_Baseload_MW']:4.1f}MW | "
                f"BESS: {res_ems['Optimal_BESS_Energy_MWh']:5.1f}MWh ({res_ems['Annual_BESS_Cycles']:3.0f} cyc/yr) | "
                f"PPA P50/P90/CVaR: {res_ems['Annualized_P50_Bankable_MEUR']:4.2f}/{res_ems['Annualized_P90_Bankable_MEUR']:4.2f}/{res_ems['Annualized_CVaR90_Bankable_MEUR']:4.2f} M€ | "
                f"AvoidedPen: {res_ems['Crisis_Avoided_Penalty_kEUR']:5.1f}k€ | "
                f"IRR Bank/Tot: {fin['IRR_Bankable_pct']:6.2f}% / {fin['IRR_Total_pct']:5.2f}%"
            )

            if np.isclose(allow_pct, 0.10):
                crisis_df = res_ems["Hourly_Results"]["S5_Crisis_2022"]
                plot_hourly_dispatch(
                    crisis_df, "S5_Crisis_2022", test_run_name, profile, allow_pct * 100, res_ems["Annual_BESS_Cycles"]
                )

    # -------------------------------------------------------------------------
    # STEP 3: Generate All Summary Plots & Save Complete CSV
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    plot_contract_profiles_comparison(summary_df, test_name=test_run_name)
    plot_risk_metrics_comparison(summary_df, all_ems_results, test_name=test_run_name)
    plot_bess_cycling_and_utilization(summary_df, test_name=test_run_name)
    plot_penalties_and_allowances_analysis(summary_df, test_name=test_run_name)

    csv_path = project_root / "plots" / test_run_name / "summary_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"\nDone! All plots and CSV saved in: {csv_path.parent}")

if __name__ == "__main__":
    main()