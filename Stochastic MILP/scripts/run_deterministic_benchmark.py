import sys
import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

with open(project_root / "data" / "config" / "project_params.yml", "r") as f:
    cfg = yaml.safe_load(f)

import numpy as np
import pandas as pd
from src.data_loader import load_or_download_year_data
from src.stochastic_ems import solve_annual_monthly_baseload_52w
from src.finance import evaluate_project_finance
from src.plotting import plot_deterministic_benchmark_2020_2022

def main():
    sim_name = "deterministic_benchmark_2020_2022"
    pv_mwp = float(cfg["project"]["pv_capacity_mwp"])
    bess_p = float(cfg["bess_default"]["power_mw"])
    bess_e = float(cfg["bess_default"]["energy_mwh"])

    df_2020 = load_or_download_year_data(2020, project_root, pv_capacity_mwp=pv_mwp, area="DK1")
    df_2022 = load_or_download_year_data(2022, project_root, pv_capacity_mwp=pv_mwp, area="DK1")

    det_years = {
        "2020_Cannibalization": {"Y2020": df_2020},
        "2022_Energy_Crisis": {"Y2022": df_2022},
    }

    profiles = ["Merchant_Only", "08h-20h", "24/7"]
    allowance_pcts = [0.00, 0.05, 0.10, 0.15]

    rows = []
    monthly_shapes = {}
    learned_2020_contracts = {}
    oracle_2022_revs = {}

    for yr_name, scen_dict in det_years.items():
        print(f"\n================ DETERMINISTIC PERFECT FORESIGHT: {yr_name} ================")
        for profile in profiles:
            pct_list = [0.0] if profile == "Merchant_Only" else allowance_pcts
            for allow_pct in pct_list:
                res = solve_annual_monthly_baseload_52w(
                    yearly_scenarios_df=scen_dict,
                    cfg=cfg,
                    allowance_pct=allow_pct,
                    contract_profile=profile,
                    bess_power_mw=bess_p,
                    bess_energy_mwh=bess_e,
                    use_fast_lp=True,
                )
                fin = evaluate_project_finance(
                    bankable_ppa_revenue_eur=res["Realized_or_Mean_Bankable_MEUR"] * 1e6,
                    merchant_bonus_revenue_eur=res["Realized_or_Mean_Merchant_MEUR"] * 1e6,
                    pv_mw=pv_mwp,
                    bess_power_mw=bess_p,
                    bess_energy_mwh=bess_e,
                    cfg=cfg,
                )

                # Store strictly deterministic metrics (no P50/P90/CVaR columns!)
                rows.append({
                    "Year_Regime": yr_name,
                    "Contract_Profile": profile,
                    "Allowance_Pct": allow_pct * 100.0,
                    "Annual_Allowance_Hours": res["Annual_Allowance_Hours"],
                    "Mean_Annual_Baseload_MW": res["Mean_Annual_Baseload_MW"],
                    "Min_Winter_Baseload_MW": res["Min_Winter_Baseload_MW"],
                    "Max_Summer_Baseload_MW": res["Max_Summer_Baseload_MW"],
                    "Annual_BESS_Cycles": res["Annual_BESS_Cycles"],
                    "Annual_Avoided_Penalty_MEUR": res["Annual_Avoided_Penalty_MEUR"],
                    "Annual_Paid_Penalty_MEUR": res["Annual_Expected_Penalty_MEUR"],
                    "Bankable_PPA_Rev_MEUR": fin["Bankable_PPA_Rev_MEUR"],
                    "Merchant_Bonus_Rev_MEUR": fin["Merchant_Bonus_Rev_MEUR"],
                    "Total_Rev_MEUR": fin["Total_Rev_MEUR"],
                    "NPV_Bankable_MEUR": fin["NPV_Bankable_MEUR"],
                    "IRR_Bankable_pct": fin["IRR_Bankable_pct"],
                    "NPV_Total_MEUR": fin["NPV_Total_MEUR"],
                    "IRR_Total_pct": fin["IRR_Total_pct"],
                })

                print(
                    f"[{profile:13s} | Allow {allow_pct*100:2.0f}%] "
                    f"BL: {res['Mean_Annual_Baseload_MW']:4.1f}MW | "
                    f"Cycles: {res['Annual_BESS_Cycles']:3.0f}/yr | "
                    f"PPA Rev: {fin['Bankable_PPA_Rev_MEUR']:5.2f}M€ | "
                    f"Spot Bonus: {fin['Merchant_Bonus_Rev_MEUR']:5.2f}M€ | "
                    f"IRR Bank/Tot: {fin['IRR_Bankable_pct']:6.2f}% / {fin['IRR_Total_pct']:5.2f}%"
                )

                if profile == "08h-20h" and allow_pct in [0.0, 0.10]:
                    monthly_shapes[f"{yr_name} (08h-20h, {int(allow_pct*100)}% Allow)"] = res["Monthly_Baseload_Profile_MW"]

                if profile == "08h-20h":
                    if yr_name == "2020_Cannibalization":
                        learned_2020_contracts[allow_pct] = res["Monthly_Baseload_Profile_MW"]
                    else:
                        oracle_2022_revs[allow_pct] = fin["Bankable_PPA_Rev_MEUR"]

    # Out-of-Sample Stress Test: What happens when the aggressive 2020 contract faces the 2022 Crisis?
    print("\n================ OUT-OF-SAMPLE CRASH TEST (2020 CONTRACT IN 2022 CRISIS) ================")
    oos_rows = []
    for allow_pct in allowance_pcts:
        res_oos = solve_annual_monthly_baseload_52w(
            yearly_scenarios_df={"Y2022": df_2022},
            cfg=cfg,
            allowance_pct=allow_pct,
            contract_profile="08h-20h",
            bess_power_mw=bess_p,
            bess_energy_mwh=bess_e,
            fixed_monthly_baseloads=learned_2020_contracts[allow_pct],
            use_fast_lp=True,
        )
        print(
            f"Allow {allow_pct*100:2.0f}% | Naive 2020 Contract in 2022 -> "
            f"Net PPA Rev: {res_oos['Realized_or_Mean_Bankable_MEUR']:6.2f} M€ "
            f"(Paid Penalties: {res_oos['Annual_Expected_Penalty_MEUR']:5.2f} M€) vs 2022 Oracle: {oracle_2022_revs[allow_pct]:5.2f} M€"
        )
        oos_rows.append({
            "Allowance_Pct": f"{int(allow_pct*100)}%",
            "Contract_Strategy": "Naive 2020 Contract (Tested in 2022)",
            "Realized_Bankable_MEUR": res_oos["Realized_or_Mean_Bankable_MEUR"],
        })
        oos_rows.append({
            "Allowance_Pct": f"{int(allow_pct*100)}%",
            "Contract_Strategy": "2022 Perfect Foresight Oracle",
            "Realized_Bankable_MEUR": oracle_2022_revs[allow_pct],
        })

    df_det = pd.DataFrame(rows)
    df_oos = pd.DataFrame(oos_rows)
    plot_deterministic_benchmark_2020_2022(df_det, monthly_shapes, df_oos, test_name=sim_name)

    csv_path = project_root / "plots" / sim_name / "deterministic_benchmark_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_det.to_csv(csv_path, index=False)
    print(f"\nAll Deterministic Benchmark plots & CSV saved in: plots/{sim_name}/")

if __name__ == "__main__":
    main()