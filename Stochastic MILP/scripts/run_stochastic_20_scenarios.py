import sys
import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

with open(project_root / "data" / "config" / "project_params.yml", "r") as f:
    cfg = yaml.safe_load(f)

import numpy as np
import pandas as pd
from src.data_loader import load_or_download_year_data, generate_20_saa_scenarios
from src.stochastic_ems import solve_annual_monthly_baseload_52w
from src.finance import evaluate_project_finance
from src.plotting import plot_stochastic_20saa_suite

def main():
    sim_name = "stochastic_20_scenarios_saa"
    pv_mwp = float(cfg["project"]["pv_capacity_mwp"])
    bess_p = float(cfg["bess_default"]["power_mw"])
    bess_e = float(cfg["bess_default"]["energy_mwh"])

    # 1. Load 2020 and 2022 historical series & build S=20 equiprobable 8760h SAA scenarios
    df_2020 = load_or_download_year_data(2020, project_root, pv_capacity_mwp=pv_mwp, area="DK1")
    df_2022 = load_or_download_year_data(2022, project_root, pv_capacity_mwp=pv_mwp, area="DK1")
    saa_20_scenarios = generate_20_saa_scenarios(df_2020, df_2022, n_scenarios=20, seed=123)
    print(f"\n[SAA Generator] Created {len(saa_20_scenarios)} equiprobable 8760h scenarios (5% probability each).")

    profiles = ["Merchant_Only", "08h-20h", "24/7"]
    allowance_pcts = [0.00, 0.05, 0.10, 0.15]

    rows = []
    scen_dist_for_plot = {}
    monthly_shapes_saa = {}

    for profile in profiles:
        print(f"\n================ TWO-STAGE STOCHASTIC (20 SAA SCENARIOS) | PROFILE: {profile} ================")
        pct_list = [0.0] if profile == "Merchant_Only" else allowance_pcts
        for allow_pct in pct_list:
            res = solve_annual_monthly_baseload_52w(
                yearly_scenarios_df=saa_20_scenarios,
                cfg=cfg,
                allowance_pct=allow_pct,
                contract_profile=profile,
                bess_power_mw=bess_p,
                bess_energy_mwh=bess_e,
                use_fast_lp=True,
            )

            # Banks evaluate Project Finance on the P90 (10th percentile) revenue across the 20 scenarios!
            fin = evaluate_project_finance(
                bankable_ppa_revenue_eur=res["Annualized_P90_Bankable_MEUR"] * 1e6,
                merchant_bonus_revenue_eur=res["Annualized_P90_Merchant_MEUR"] * 1e6,
                pv_mw=pv_mwp,
                bess_power_mw=bess_p,
                bess_energy_mwh=bess_e,
                cfg=cfg,
            )

            row = {k: v for k, v in {**res, **fin}.items() if not isinstance(v, dict)}
            rows.append(row)

            print(
                f"Allow {allow_pct*100:2.0f}% | Stage-1 BL: {res['Mean_Annual_Baseload_MW']:4.1f}MW | "
                f"PPA P50 / P90 / VaR90 / CVaR90: "
                f"{res['Annualized_P50_Bankable_MEUR']:4.2f} / {res['Annualized_P90_Bankable_MEUR']:4.2f} / "
                f"{res['Annualized_VaR90_Bankable_MEUR']:4.2f} / {res['Annualized_CVaR90_Bankable_MEUR']:4.2f} M€ | "
                f"AvoidedPen: {res['Annual_Avoided_Penalty_MEUR']:4.2f}M€ | "
                f"P90 IRR Bank/Tot: {fin['IRR_Bankable_pct']:6.2f}% / {fin['IRR_Total_pct']:5.2f}%"
            )

            if profile == "08h-20h" and allow_pct in [0.0, 0.10]:
                scen_dist_for_plot[f"08h-20h ({int(allow_pct*100)}% Allow)"] = res["Scenario_Bankable_MEUR"]
                monthly_shapes_saa[f"08h-20h ({int(allow_pct*100)}% Allow)"] = res["Monthly_Baseload_Profile_MW"]
            if profile == "24/7" and allow_pct in [0.0, 0.10]:
                monthly_shapes_saa[f"24/7 ({int(allow_pct*100)}% Allow)"] = res["Monthly_Baseload_Profile_MW"]

    df_stoch = pd.DataFrame(rows)
    plot_stochastic_20saa_suite(df_stoch, scen_dist_for_plot, monthly_shapes_saa, test_name=sim_name)

    csv_path = project_root / "plots" / sim_name / "stochastic_20saa_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_stoch.to_csv(csv_path, index=False)
    print(f"\nAll Two-Stage Stochastic (20 SAA) plots & CSV saved in: plots/{sim_name}/")

if __name__ == "__main__":
    main()