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
from src.stochastic_ems import solve_full_year_8760h_coopt
from src.finance import evaluate_project_finance
from src.plotting import plot_deterministic_benchmark_2020_2022

def main():
    sim_name = "deterministic_benchmark_2020_2022"
    pv_mwp = float(cfg["project"]["pv_capacity_mwp"])

    # 1. Load 8760-hour historical datasets for 2020 and 2022
    df_2020 = load_or_download_year_data(2020, project_root, pv_capacity_mwp=pv_mwp, area="DK1")
    df_2022 = load_or_download_year_data(2022, project_root, pv_capacity_mwp=pv_mwp, area="DK1")

    det_years = {
        "2020_Cannibalization": df_2020,
        "2022_Energy_Crisis": df_2022,
    }

    profiles = ["Merchant_Only", "24/7", "08h-20h", "12h-00h"]
    allowance_pcts = [0.00, 0.05, 0.10, 0.15]

    in_sample_rows = []
    monthly_shapes_records = []
    learned_designs = {}  # Stores {(yr_name, profile, allow_pct): full result dict}

    # =========================================================================
    # STEP 1: IN-SAMPLE PERFECT FORESIGHT CO-OPTIMIZATION (BESS MW/MWh + 12 B_m)
    # =========================================================================
    for yr_name, df_yr in det_years.items():
        print(f"\n================ IN-SAMPLE CO-OPTIMIZATION (BESS + 12 MONTHLY BL): {yr_name} ================")
        for profile in profiles:
            pct_list = [0.0] if profile == "Merchant_Only" else allowance_pcts
            for allow_pct in pct_list:
                res = solve_full_year_8760h_coopt(
                    df_year=df_yr,
                    cfg=cfg,
                    allowance_pct=allow_pct,
                    contract_profile=profile,
                    optimize_bess_size=True,  # Co-optimizes BESS Power (MW) & Storage (MWh) + 12 B_m!
                )
                fin = evaluate_project_finance(
                    bankable_ppa_revenue_eur=res["Realized_Bankable_MEUR"] * 1e6,
                    merchant_bonus_revenue_eur=res["Realized_Merchant_MEUR"] * 1e6,
                    pv_mw=pv_mwp,
                    bess_power_mw=res["Optimal_BESS_Power_MW"],
                    bess_energy_mwh=res["Optimal_BESS_Energy_MWh"],
                    cfg=cfg,
                )

                learned_designs[(yr_name, profile, allow_pct)] = {**res, **fin}

                row = {
                    "Year_Regime": yr_name,
                    "Contract_Profile": profile,
                    "Allowance_Pct": allow_pct * 100.0,
                    "Annual_Allowance_Hours": res["Annual_Allowance_Hours"],
                    "Mean_Annual_Baseload_MW": res["Mean_Annual_Baseload_MW"],
                    "Min_Winter_Baseload_MW": res["Min_Winter_Baseload_MW"],
                    "Max_Summer_Baseload_MW": res["Max_Summer_Baseload_MW"],
                    "Optimal_BESS_Power_MW": res["Optimal_BESS_Power_MW"],
                    "Optimal_BESS_Energy_MWh": res["Optimal_BESS_Energy_MWh"],
                    "Annual_BESS_Cycles": res["Annual_BESS_Cycles"],
                    "Annual_Avoided_Penalty_MEUR": res["Annual_Avoided_Penalty_MEUR"],
                    "Annual_Paid_Penalty_MEUR": res["Annual_Paid_Penalty_MEUR"],
                    "CAPEX_Total_MEUR": fin["CAPEX_Total_MEUR"],
                    "CAPEX_BESS_MEUR": fin["CAPEX_BESS_MEUR"],
                    "Bankable_PPA_Rev_MEUR": fin["Bankable_PPA_Rev_MEUR"],
                    "Merchant_Bonus_Rev_MEUR": fin["Merchant_Bonus_Rev_MEUR"],
                    "Total_Rev_MEUR": fin["Total_Rev_MEUR"],
                    "NPV_Bankable_MEUR": fin["NPV_Bankable_MEUR"],
                    "IRR_Bankable_pct": fin["IRR_Bankable_pct"],
                    "NPV_Total_MEUR": fin["NPV_Total_MEUR"],
                    "IRR_Total_pct": fin["IRR_Total_pct"],
                }
                in_sample_rows.append(row)

                if profile != "Merchant_Only":
                    for m_idx, bl_mw in res["Monthly_Baseload_Profile_MW"].items():
                        monthly_shapes_records.append({
                            "Year_Regime": yr_name,
                            "Contract_Profile": profile,
                            "Allowance_Pct": allow_pct * 100.0,
                            "Month": m_idx,
                            "Baseload_MW": bl_mw,
                        })

                print(
                    f"[{profile:13s} | Allow {allow_pct*100:2.0f}%] "
                    f"Mean BL: {res['Mean_Annual_Baseload_MW']:4.1f}MW | "
                    f"BESS: {res['Optimal_BESS_Power_MW']:4.1f}MW / {res['Optimal_BESS_Energy_MWh']:5.1f}MWh ({res['Annual_BESS_Cycles']:3.0f} cyc) | "
                    f"PPA Rev: {fin['Bankable_PPA_Rev_MEUR']:5.2f}M€ | "
                    f"Bonus: {fin['Merchant_Bonus_Rev_MEUR']:5.2f}M€ | "
                    f"IRR Bank/Tot: {fin['IRR_Bankable_pct']:6.2f}% / {fin['IRR_Total_pct']:5.2f}%"
                )

    # =========================================================================
    # STEP 2: CROSS-SCENARIO OUT-OF-SAMPLE EVALUATION FOR ALL COMBINATIONS
    # (1) Trained on 2020 -> Tested on 2022
    # (2) Trained on 2022 -> Tested on 2020
    # =========================================================================
    cross_pairs = [
        ("2020_Cannibalization", "2022_Energy_Crisis", df_2022, "Trained_2020_Tested_in_2022"),
        ("2022_Energy_Crisis", "2020_Cannibalization", df_2020, "Trained_2022_Tested_in_2020"),
    ]

    cross_rows = []
    for train_yr, test_yr, df_test, dir_label in cross_pairs:
        print(f"\n================ CROSS-EVALUATION: {dir_label} ================")
        for profile in profiles:
            pct_list = [0.0] if profile == "Merchant_Only" else allowance_pcts
            for allow_pct in pct_list:
                design_train = learned_designs[(train_yr, profile, allow_pct)]
                native_oracle = learned_designs[(test_yr, profile, allow_pct)]

                # Solve deterministic dispatch on `df_test` with BESS & Monthly Baseloads fixed from `train_yr`
                res_oos = solve_full_year_8760h_coopt(
                    df_year=df_test,
                    cfg=cfg,
                    allowance_pct=allow_pct,
                    contract_profile=profile,
                    optimize_bess_size=False,
                    fixed_bess_power_mw=design_train["Optimal_BESS_Power_MW"],
                    fixed_bess_energy_mwh=design_train["Optimal_BESS_Energy_MWh"],
                    fixed_monthly_baseloads=design_train["Monthly_Baseload_Profile_MW"],
                )

                fin_oos = evaluate_project_finance(
                    bankable_ppa_revenue_eur=res_oos["Realized_Bankable_MEUR"] * 1e6,
                    merchant_bonus_revenue_eur=res_oos["Realized_Merchant_MEUR"] * 1e6,
                    pv_mw=pv_mwp,
                    bess_power_mw=design_train["Optimal_BESS_Power_MW"],
                    bess_energy_mwh=design_train["Optimal_BESS_Energy_MWh"],
                    cfg=cfg,
                )

                cross_rows.append({
                    "Evaluation_Direction": dir_label,
                    "Trained_On": train_yr,
                    "Tested_On": test_yr,
                    "Contract_Profile": profile,
                    "Allowance_Pct": allow_pct * 100.0,
                    "Fixed_Mean_Baseload_MW": design_train["Mean_Annual_Baseload_MW"],
                    "Fixed_BESS_Power_MW": design_train["Optimal_BESS_Power_MW"],
                    "Fixed_BESS_Energy_MWh": design_train["Optimal_BESS_Energy_MWh"],
                    "OOS_Paid_Penalty_MEUR": res_oos["Annual_Paid_Penalty_MEUR"],
                    "OOS_Avoided_Penalty_MEUR": res_oos["Annual_Avoided_Penalty_MEUR"],
                    "OOS_Bankable_PPA_Rev_MEUR": fin_oos["Bankable_PPA_Rev_MEUR"],
                    "OOS_Merchant_Bonus_Rev_MEUR": fin_oos["Merchant_Bonus_Rev_MEUR"],
                    "OOS_Total_Rev_MEUR": fin_oos["Total_Rev_MEUR"],
                    "OOS_IRR_Bankable_pct": fin_oos["IRR_Bankable_pct"],
                    "OOS_IRR_Total_pct": fin_oos["IRR_Total_pct"],
                    "Native_Oracle_Bankable_Rev_MEUR": native_oracle["Bankable_PPA_Rev_MEUR"],
                    "Native_Oracle_IRR_Bankable_pct": native_oracle["IRR_Bankable_pct"],
                    "Native_Oracle_IRR_Total_pct": native_oracle["IRR_Total_pct"],
                })

                print(
                    f"[{profile:13s} | Allow {allow_pct*100:2.0f}%] "
                    f"Fixed BL: {design_train['Mean_Annual_Baseload_MW']:4.1f}MW & BESS: {design_train['Optimal_BESS_Energy_MWh']:5.1f}MWh | "
                    f"OOS PPA Rev: {fin_oos['Bankable_PPA_Rev_MEUR']:6.2f}M€ (Pen: {res_oos['Annual_Paid_Penalty_MEUR']:5.2f}M€) | "
                    f"OOS IRR Tot: {fin_oos['IRR_Total_pct']:6.2f}% vs Oracle: {native_oracle['IRR_Total_pct']:5.2f}%"
                )

    # =========================================================================
    # STEP 3: SAVE ALL PLOTS AND CSV TABLES
    # =========================================================================
    df_det = pd.DataFrame(in_sample_rows)
    df_cross = pd.DataFrame(cross_rows)

    plot_deterministic_benchmark_2020_2022(df_det, monthly_shapes_records, df_cross, test_name=sim_name)

    out_dir = project_root / "plots" / sim_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df_det.to_csv(out_dir / "deterministic_in_sample_summary.csv", index=False)
    df_cross.to_csv(out_dir / "deterministic_cross_scenario_oos.csv", index=False)
    pd.DataFrame(monthly_shapes_records).to_csv(out_dir / "deterministic_monthly_baseload_shapes.csv", index=False)

    print(f"\nDone! All 4 deterministic plot suites and 3 CSV tables saved in: {out_dir}")

if __name__ == "__main__":
    main()