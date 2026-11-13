import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid")

def _get_plot_dir(test_name: str) -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    plot_dir = root_dir / "plots" / test_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def plot_hourly_dispatch(hourly_df: pd.DataFrame, scenario_name: str, test_name: str, profile: str, allow_pct: float, annual_cycles: float = 0.0):
    """
    1. Existing Hourly Dispatch Plot (upgraded with BESS Cycles, Free Allowance & Penalized Shortfall).
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    safe_prof = profile.replace("/", "-")
    fig.suptitle(
        f"Dispatch ({profile}) - {scenario_name} | Allowances: {allow_pct:.0f}% | BESS Cycling: {annual_cycles:.0f} cycles/yr",
        fontsize=14, fontweight='bold'
    )
    t_idx = hourly_df.index

    # 1. Solar Allocation
    axs[0].plot(t_idx, hourly_df['pv_avail'], label='PV Available', color='orange', linestyle='--', alpha=0.6)
    axs[0].fill_between(t_idx, 0, hourly_df['pv_to_grid'], label='PV to Grid', color='gold', alpha=0.7)
    axs[0].fill_between(t_idx, hourly_df['pv_to_grid'], hourly_df['pv_to_grid'] + hourly_df['pv_to_bess'], label='PV Charging BESS', color='mediumseagreen', alpha=0.7)
    axs[0].set_ylabel("MW")
    axs[0].set_title("1. Solar PV Generation & PV-Only Battery Charging")
    axs[0].legend(loc="upper right")

    # 2. PPA Delivery vs Merchant Bonus vs Free Allowance vs Penalized Shortfall
    axs[1].fill_between(t_idx, 0, hourly_df['q_ppa_deliv'], label='Firm PPA Delivered (75 €/MWh)', color='royalblue', alpha=0.8)
    axs[1].fill_between(t_idx, hourly_df['q_ppa_deliv'], hourly_df['q_ppa_deliv'] + hourly_df['q_merchant_bonus'], label='Merchant Grid Sales (Spot)', color='darkorange', alpha=0.65)
    axs[1].bar(t_idx, hourly_df['q_allow_free'], label='Free Allowance Used (0 € Penalty)', color='limegreen', alpha=0.85)
    axs[1].bar(t_idx, hourly_df['q_excess_shortfall'], bottom=hourly_df['q_allow_free'], label='Excess Shortfall (PENALIZED!)', color='crimson', alpha=0.9)
    axs[1].set_ylabel("MW")
    axs[1].set_title(f"2. Power Split ({profile}): Delivered vs Direct Market Sales vs Allowances/Shortfall")
    axs[1].legend(loc="upper right")

    # 3. Spot Price & Allowance / Penalty Triggers
    axs[2].plot(t_idx, hourly_df['price_da'], label='Spot Price DA (€/MWh)', color='purple', linewidth=1.8)
    allow_idx = hourly_df[hourly_df['u_allow'] == 1].index
    for idx, ah in enumerate(allow_idx):
        axs[2].axvspan(ah - 0.5, ah + 0.5, color='limegreen', alpha=0.25, label='Free Allowance Window' if idx == 0 else "")
    exc_idx = hourly_df[hourly_df['q_excess_shortfall'] > 0.01].index
    for idx, eh in enumerate(exc_idx):
        axs[2].axvspan(eh - 0.5, eh + 0.5, color='crimson', alpha=0.35, label='Penalized Default Hour' if idx == 0 else "")
    axs[2].set_ylabel("€/MWh")
    axs[2].set_title("3. Spot Price, Free Allowance Windows (Green) & Penalized Default Hours (Red)")
    axs[2].legend(loc="upper right")

    # 4. BESS Discharge & SoC
    ax4_twin = axs[3].twinx()
    axs[3].bar(t_idx, hourly_df['bess_to_grid'], label='BESS Discharge (MW)', color='steelblue', alpha=0.5)
    ax4_twin.plot(t_idx, hourly_df['soc_mwh'], label='State of Charge (MWh)', color='teal', linewidth=2)
    axs[3].set_ylabel("Discharge (MW)", color='steelblue')
    ax4_twin.set_ylabel("SoC (MWh)", color='teal')
    axs[3].set_xlabel("Hour of Week")
    axs[3].set_title(f"4. Battery Discharge & SoC (Equivalent Full Cycles: {annual_cycles:.1f} cycles/yr)")

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / f"dispatch_{safe_prof}_allow_{int(allow_pct)}pct.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_contract_profiles_comparison(summary_df: pd.DataFrame, test_name: str):
    """
    2. Existing Benchmark Plot (upgraded to include 'Merchant_Only' horizontal reference line).
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Contract Profile & Allowance Benchmark vs Direct Market Sale (Merchant Only)", fontsize=15, fontweight='bold')

    ppa_df = summary_df[summary_df["Contract_Profile"] != "Merchant_Only"]
    merch_df = summary_df[summary_df["Contract_Profile"] == "Merchant_Only"]

    # Panel 1: Optimal Baseload MW
    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Optimal_Baseload_MW", hue="Contract_Profile", marker="o", linewidth=2.5, ax=axs[0])
    axs[0].axhline(0.0, color="darkorange", linestyle="--", linewidth=2, label="Merchant_Only (0 MW Baseload)")
    axs[0].set_title("1. Achievable Firm Baseload (MW)")
    axs[0].set_xlabel("Free Allowance Quota (% of Active Hours)")
    axs[0].set_ylabel("Optimal Baseload Promised (MW)")
    axs[0].legend()

    # Panel 2: Bankable IRR (Solid) vs Total IRR (Dashed) + Merchant_Only baseline
    for prof, grp in ppa_df.groupby("Contract_Profile"):
        line = axs[1].plot(grp["Allowance_Pct"], grp["IRR_Bankable_pct"], marker="o", linewidth=2, label=f"{prof} (Bankable PPA)")
        axs[1].plot(grp["Allowance_Pct"], grp["IRR_Total_pct"], marker="s", linewidth=2, linestyle="--", color=line[0].get_color(), alpha=0.75, label=f"{prof} (Total + Spot)")
    if not merch_df.empty:
        merch_irr = merch_df["IRR_Total_pct"].iloc[0]
        axs[1].axhline(merch_irr, color="crimson", linestyle="-.", linewidth=2.2, label=f"Merchant_Only IRR ({merch_irr:.2f}%)")
    axs[1].axhline(6.5, color="black", linestyle=":", label="WACC Target (6.5%)")
    axs[1].set_title("2. Bankable IRR (Solid) vs Total IRR (Dashed) vs Merchant")
    axs[1].set_xlabel("Allowance Quota (%)")
    axs[1].set_ylabel("Project IRR (%)")
    axs[1].legend(fontsize=7.5, loc="best")

    # Panel 3: Bankable NPV vs Total NPV @ 10% Allowance (including Merchant_Only)
    sub_ppa = ppa_df[np.isclose(ppa_df["Allowance_Pct"], 10.0)]
    plot_sub = pd.concat([merch_df.iloc[:1], sub_ppa], ignore_index=True) if not merch_df.empty else sub_ppa
    x = np.arange(len(plot_sub))
    width = 0.35
    axs[2].bar(x - width/2, plot_sub["NPV_Bankable_MEUR"], width, label="Bankable NPV (PPA Only)", color="royalblue")
    axs[2].bar(x + width/2, plot_sub["NPV_Total_MEUR"], width, label="Total NPV (PPA + Spot Sales)", color="darkorange")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(plot_sub["Contract_Profile"], rotation=15)
    axs[2].axhline(0, color="black", linewidth=1)
    axs[2].set_title("3. NPV Comparison @ 10% Allowances vs Merchant_Only")
    axs[2].set_ylabel("NPV (M€)")
    axs[2].legend()

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / "benchmark_profiles_bankable_vs_bonus.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {file_path}")

def plot_risk_metrics_comparison(summary_df: pd.DataFrame, all_ems_results: dict, test_name: str):
    """
    3. NEW PLOT: Explicit comparison of P50, P90, VaR 90, and CVaR 90 across Scenarios & Contract Profiles.
    """
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Stochastic Risk Analysis: P50, P90, VaR 90 & CVaR 90 Across Scenarios", fontsize=16, fontweight='bold')

    ppa_df = summary_df[summary_df["Contract_Profile"] != "Merchant_Only"]
    merch_df = summary_df[summary_df["Contract_Profile"] == "Merchant_Only"]

    # Panel 1: Bankable Revenue Risk Metrics (P50, P90, VaR90, CVaR90) for best profile (08h-20h)
    prof_focus = "08h-20h" if "08h-20h" in ppa_df["Contract_Profile"].values else ppa_df["Contract_Profile"].iloc[0]
    df_f = ppa_df[ppa_df["Contract_Profile"] == prof_focus]
    axs[0].plot(df_f["Allowance_Pct"], df_f["Annualized_P50_Bankable_MEUR"], marker="o", linewidth=2.5, color="navy", label="P50 (Median)")
    axs[0].plot(df_f["Allowance_Pct"], df_f["Annualized_P90_Bankable_MEUR"], marker="s", linewidth=2.2, color="dodgerblue", label="P90 (10th Percentile)")
    axs[0].plot(df_f["Allowance_Pct"], df_f["Annualized_VaR90_Bankable_MEUR"], marker="^", linestyle="--", linewidth=2, color="darkorange", label="VaR 90 Threshold")
    axs[0].plot(df_f["Allowance_Pct"], df_f["Annualized_CVaR90_Bankable_MEUR"], marker="d", linestyle=":", linewidth=2.5, color="crimson", label="CVaR 90 (Worst 10% Tail)")
    axs[0].set_title(f"1. Bankable PPA Revenue Risk Metrics ({prof_focus})")
    axs[0].set_xlabel("Allowance Quota (%)")
    axs[0].set_ylabel("Annualized Bankable Revenue (M€/yr)")
    axs[0].legend()

    # Panel 2: Total Revenue (PPA + Spot) P50, P90, VaR90, CVaR90 at 10% Allowance across ALL profiles (incl. Merchant_Only)
    sub_ppa = ppa_df[np.isclose(ppa_df["Allowance_Pct"], 10.0)]
    comp_df = pd.concat([merch_df.iloc[:1], sub_ppa], ignore_index=True) if not merch_df.empty else sub_ppa
    x = np.arange(len(comp_df))
    w_bar = 0.20
    axs[1].bar(x - 1.5*w_bar, comp_df["Annualized_P50_Total_MEUR"], w_bar, label="P50 (Median)", color="navy")
    axs[1].bar(x - 0.5*w_bar, comp_df["Annualized_P90_Total_MEUR"], w_bar, label="P90 (10th Pct)", color="royalblue")
    axs[1].bar(x + 0.5*w_bar, comp_df["Annualized_VaR90_Total_MEUR"], w_bar, label="VaR 90 (Quantile)", color="goldenrod")
    axs[1].bar(x + 1.5*w_bar, comp_df["Annualized_CVaR90_Total_MEUR"], w_bar, label="CVaR 90 (Tail Mean)", color="crimson")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(comp_df["Contract_Profile"], rotation=15)
    axs[1].set_title("2. Total Revenue (P50 / P90 / VaR90 / CVaR90) @ 10% Allowance")
    axs[1].set_ylabel("Annualized Total Revenue (M€/yr)")
    axs[1].legend()

    # Panel 3: Scenario-by-Scenario Revenue Distribution for 08h-20h (10% Allow) vs Merchant_Only
    scen_rows = []
    for key_label, res_key in [("Merchant_Only", ("Merchant_Only", 0.0)), (f"{prof_focus} (10% Allow)", (prof_focus, 0.10))]:
        if res_key in all_ems_results:
            res_obj = all_ems_results[res_key]
            for scen_name, rev_eur in res_obj["Scenario_Total_EUR"].items():
                scen_rows.append({"Strategy": key_label, "Scenario": scen_name, "Total_Rev_MEUR": rev_eur / 1e6})
    if scen_rows:
        df_scen = pd.DataFrame(scen_rows)
        sns.barplot(data=df_scen, x="Scenario", y="Total_Rev_MEUR", hue="Strategy", palette=["darkorange", "royalblue"], ax=axs[2])
        axs[2].tick_params(axis='x', rotation=25)
        axs[2].set_title(f"3. Scenario Revenues: Direct Market vs {prof_focus} PPA")
        axs[2].set_ylabel("Total Annualized Revenue (M€/yr)")

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / "risk_metrics_p50_p90_var_cvar.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {file_path}")

def plot_bess_cycling_and_utilization(summary_df: pd.DataFrame, test_name: str):
    """
    4. NEW PLOT: Battery Cycling (Equivalent Full Cycles/yr), Optimal Capacity (MWh) & Degradation Cost.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Battery Utilization Analysis: Equivalent Full Cycles (EFC), Sizing & Degradation", fontsize=15, fontweight='bold')

    ppa_df = summary_df[summary_df["Contract_Profile"] != "Merchant_Only"]
    merch_df = summary_df[summary_df["Contract_Profile"] == "Merchant_Only"]

    # Panel 1: Annual Equivalent Full Cycles (cycles/yr) vs Allowance %
    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Annual_BESS_Cycles", hue="Contract_Profile", marker="o", linewidth=2.5, ax=axs[0])
    if not merch_df.empty:
        m_cyc = merch_df["Annual_BESS_Cycles"].iloc[0]
        axs[0].axhline(m_cyc, color="crimson", linestyle="--", linewidth=2, label=f"Merchant_Only ({m_cyc:.0f} cyc/yr)")
    axs[0].set_title("1. Battery Cycling Intensity (Equivalent Full Cycles / yr)")
    axs[0].set_xlabel("Allowance Quota (%)")
    axs[0].set_ylabel("Annual BESS Cycles (EFC/yr)")
    axs[0].legend()

    # Panel 2: Optimal BESS Energy Capacity (MWh) vs Allowance %
    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Optimal_BESS_Energy_MWh", hue="Contract_Profile", marker="s", linewidth=2.5, ax=axs[1])
    if not merch_df.empty:
        m_e = merch_df["Optimal_BESS_Energy_MWh"].iloc[0]
        axs[1].axhline(m_e, color="crimson", linestyle="--", linewidth=2, label=f"Merchant_Only ({m_e:.1f} MWh)")
    axs[1].set_title("2. Optimal Battery Storage Size (MWh)")
    axs[1].set_xlabel("Allowance Quota (%)")
    axs[1].set_ylabel("Optimal BESS Capacity (MWh)")
    axs[1].legend()

    # Panel 3: Annual Energy Discharged (GWh/yr) & Degradation Cost (M€/yr) @ 10% Allowance
    sub_ppa = ppa_df[np.isclose(ppa_df["Allowance_Pct"], 10.0)]
    comp_df = pd.concat([merch_df.iloc[:1], sub_ppa], ignore_index=True) if not merch_df.empty else sub_ppa
    x = np.arange(len(comp_df))
    ax3_twin = axs[2].twinx()
    b1 = axs[2].bar(x - 0.18, comp_df["Annual_BESS_Discharged_MWh"] / 1e3, width=0.35, color="teal", alpha=0.8, label="BESS Throughput (GWh/yr)")
    b2 = ax3_twin.bar(x + 0.18, comp_df["Annual_Degradation_Cost_MEUR"], width=0.35, color="indianred", alpha=0.8, label="Cycling Degradation Cost (M€/yr)")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(comp_df["Contract_Profile"], rotation=15)
    axs[2].set_ylabel("Annual Energy Discharged (GWh/yr)", color="teal")
    ax3_twin.set_ylabel("Annual Degradation Cost (M€/yr)", color="indianred")
    axs[2].set_title("3. BESS Throughput & Cycling Wear Cost @ 10% Allowance")

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / "bess_utilization_and_cycling.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {file_path}")

def plot_penalties_and_allowances_analysis(summary_df: pd.DataFrame, test_name: str):
    """
    5. NEW PLOT: Comprehensive Penalty & Allowance Analysis across Contract Profiles.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Allowance & Penalty Analysis: Free Quota Usage vs Penalized Excess Shortfall", fontsize=15, fontweight='bold')

    ppa_df = summary_df[summary_df["Contract_Profile"] != "Merchant_Only"]

    # Panel 1: Crisis Free Allowance MWh Used vs Allowance Quota (%)
    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Crisis_Free_Allowance_MWh", hue="Contract_Profile", marker="o", linewidth=2.5, ax=axs[0])
    axs[0].set_title("1. Free Allowance Energy Used in Crisis (MWh/week)")
    axs[0].set_xlabel("Free Allowance Quota (%)")
    axs[0].set_ylabel("Free Allowance Volume (MWh)")

    # Panel 2: Avoided Spot Penalty Thanks to Free Allowances (k€ in Crisis Week)
    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Crisis_Avoided_Penalty_kEUR", hue="Contract_Profile", marker="s", linewidth=2.5, ax=axs[1])
    axs[1].set_title("2. Spot Penalties Avoided by Free Allowances in Crisis (k€)")
    axs[1].set_xlabel("Free Allowance Quota (%)")
    axs[1].set_ylabel("Avoided Penalty Cost (k€ / crisis week)")

    # Panel 3: Stacked Comparison of Free Allowance vs Penalized Excess Shortfall & Paid Penalty
    sub_df = ppa_df[ppa_df["Allowance_Pct"].isin([5.0, 10.0, 15.0])]
    sns.barplot(data=sub_df, x="Contract_Profile", y="Annual_Avoided_Penalty_MEUR", hue="Allowance_Pct", palette="Greens_d", ax=axs[2])
    axs[2].set_title("3. Annualized Value of Avoided Penalties (M€/yr)")
    axs[2].set_xlabel("Contract Profile")
    axs[2].set_ylabel("Annual Avoided Penalty (M€/yr)")
    axs[2].legend(title="Allowance %")

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / "penalties_and_allowances_analysis.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {file_path}")

def plot_backtest_2020_vs_2022(df_backtest: pd.DataFrame, monthly_shapes: dict, weekly_df: pd.DataFrame, test_name: str):
    """
    6. NEW PLOT: Full 52-Week / 12-Month Backtest Comparison between 2020 (Low Price / Cannibalization),
    2022 (Energy Crisis), and Joint Stochastic Sizing (2020 + 2022).
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("52-Week Backtest (8760h): 2020 (Cannibalization) vs 2022 (Energy Crisis) vs Joint Stochastic", fontsize=16, fontweight='bold')

    # Panel 1: Monthly Baseload Shaping (Jan - Dec) across Allowance Levels
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for label, shape_dict in monthly_shapes.items():
        vals = [shape_dict[m] for m in months]
        axs[0, 0].plot(months, vals, marker="o", linewidth=2.2, label=label)
    axs[0, 0].set_xticks(months)
    axs[0, 0].set_xticklabels(month_labels)
    axs[0, 0].set_title("1. Optimal Monthly Baseload Shaping Across the Year (Jan - Dec)")
    axs[0, 0].set_ylabel("Monthly Promised Baseload B_m (MW)")
    axs[0, 0].legend(fontsize=8.5)

    # Panel 2: Bankable IRR vs Total IRR across Regimes (2020 vs 2022 vs Joint)
    ppa_only = df_backtest[df_backtest["Contract_Profile"] != "Merchant_Only"]
    sns.lineplot(data=ppa_only, x="Allowance_Pct", y="IRR_Total_pct", hue="Market_Regime", style="Contract_Profile", markers=True, linewidth=2.2, ax=axs[0, 1])
    axs[0, 1].axhline(6.5, color="black", linestyle=":", label="WACC (6.5%)")
    axs[0, 1].set_title("2. Total Project IRR (%) across Market Regimes (2020 vs 2022)")
    axs[0, 1].set_xlabel("Allowance Quota (%)")
    axs[0, 1].set_ylabel("Total Project IRR (%)")
    axs[0, 1].legend(fontsize=8)

    # Panel 3: Avoided Spot Penalties in 2020 vs 2022 (M€/yr)
    sns.barplot(data=ppa_only, x="Allowance_Pct", y="Annual_Avoided_Penalty_MEUR", hue="Market_Regime", palette="Set2", ax=axs[1, 0])
    axs[1, 0].set_title("3. Annual Spot Penalties Avoided by Free Allowances (2020 vs 2022)")
    axs[1, 0].set_xlabel("Free Allowance Quota (%)")
    axs[1, 0].set_ylabel("Avoided Spot Penalty (M€/yr)")

    # Panel 4: 52-Week Cumulative Revenue Trajectory (2020 vs 2022)
    for col in [c for c in weekly_df.columns if c != "week"]:
        axs[1, 1].plot(weekly_df["week"], weekly_df[col], linewidth=2.2, label=col)
    axs[1, 1].set_title("4. 52-Week Cumulative Revenue Trajectory (M€)")
    axs[1, 1].set_xlabel("Week of the Year (1 to 52)")
    axs[1, 1].set_ylabel("Cumulative Annual Revenue (M€)")
    axs[1, 1].legend(fontsize=8.5)

    plt.tight_layout()
    plot_dir = _get_plot_dir(test_name)
    file_path = plot_dir / "backtest_52w_2020_vs_2022.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 52-week backtest plot: {file_path}")

def plot_deterministic_benchmark_2020_2022(
    df_det: pd.DataFrame,
    monthly_shapes: dict,
    df_oos_crash: pd.DataFrame,
    test_name: str
):
    """
    Plots for Script 1 (`deterministic_benchmark_2020_2022`):
    No P50/P90/CVaR jargon here — strictly Deterministic Perfect Foresight vs Out-of-Sample Crash Test!
    """
    plot_dir = _get_plot_dir(test_name)
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # --- PLOT 1: 4-Panel Deterministic Overview ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Deterministic Benchmark (Perfect Foresight): 2020 (Cannibalization) vs 2022 (Energy Crisis)", fontsize=15, fontweight="bold")

    # 1. Seasonal Baseload Shaping B_m (2020 vs 2022)
    for label, shape_dict in monthly_shapes.items():
        axs[0, 0].plot(months, [shape_dict[m] for m in months], marker="o", linewidth=2.2, label=label)
    axs[0, 0].set_xticks(months)
    axs[0, 0].set_xticklabels(month_labels)
    axs[0, 0].set_title("1. Perfect Foresight Monthly Baseload Shaping (MW)")
    axs[0, 0].set_ylabel("Promised Monthly Baseload B_m (MW)")
    axs[0, 0].legend(fontsize=8.5)

    # 2. Deterministic Bankable IRR vs Total IRR
    ppa_sub = df_det[df_det["Contract_Profile"] != "Merchant_Only"]
    sns.lineplot(data=ppa_sub, x="Allowance_Pct", y="IRR_Bankable_pct", hue="Year_Regime", style="Contract_Profile", markers=True, linewidth=2.2, ax=axs[0, 1])
    axs[0, 1].axhline(6.5, color="black", linestyle=":", label="WACC Target (6.5%)")
    axs[0, 1].set_title("2. Deterministic Bankable IRR (PPA Revenue Only)")
    axs[0, 1].set_xlabel("Free Allowance Quota (%)")
    axs[0, 1].set_ylabel("Deterministic Bankable IRR (%)")

    # 3. Avoided Spot Penalties in 2020 vs 2022
    sns.barplot(data=ppa_sub, x="Allowance_Pct", y="Annual_Avoided_Penalty_MEUR", hue="Year_Regime", palette="Set1", ax=axs[1, 0])
    axs[1, 0].set_title("3. Annual Spot Penalties Avoided by Free Allowances (M€/yr)")
    axs[1, 0].set_xlabel("Free Allowance Quota (%)")
    axs[1, 0].set_ylabel("Avoided Spot Penalty (M€/yr)")

    # 4. Out-of-Sample Crash Test: 2020 Contract Tested in 2022 Crisis!
    sns.barplot(data=df_oos_crash, x="Allowance_Pct", y="Realized_Bankable_MEUR", hue="Contract_Strategy", palette=["crimson", "royalblue"], ax=axs[1, 1])
    axs[1, 1].axhline(0, color="black", linewidth=1)
    axs[1, 1].set_title("4. Out-of-Sample Crash Test in 2022: Naive 2020 Contract vs 2022 Oracle")
    axs[1, 1].set_xlabel("Free Allowance Quota (%)")
    axs[1, 1].set_ylabel("Realized 2022 Net PPA Revenue (M€/yr)")
    axs[1, 1].legend(fontsize=8.5)

    plt.tight_layout()
    p1 = plot_dir / "01_deterministic_2020_vs_2022_overview.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p1}")

    # --- PLOT 2: BESS Cycling & Revenue Breakdown in Deterministic Mode ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2.suptitle("Deterministic Battery Cycling & Revenue Split (2020 vs 2022)", fontsize=14, fontweight="bold")
    sns.lineplot(data=ppa_sub, x="Allowance_Pct", y="Annual_BESS_Cycles", hue="Year_Regime", style="Contract_Profile", markers=True, linewidth=2.5, ax=axs2[0])
    axs2[0].set_title("1. Annual Battery Cycling (Equivalent Full Cycles / yr)")
    axs2[0].set_ylabel("Cycles / yr")

    sub10 = df_det[df_det["Allowance_Pct"].isin([0.0, 10.0]) & (df_det["Contract_Profile"] == "08h-20h")]
    x = np.arange(len(sub10))
    labels = [f"{r['Year_Regime']}\n({int(r['Allowance_Pct'])}% Allow)" for _, r in sub10.iterrows()]
    axs2[1].bar(x, sub10["Bankable_PPA_Rev_MEUR"], width=0.45, label="Bankable PPA Revenue", color="royalblue")
    axs2[1].bar(x, sub10["Merchant_Bonus_Rev_MEUR"], width=0.45, bottom=sub10["Bankable_PPA_Rev_MEUR"], label="Merchant Spot Bonus", color="darkorange")
    axs2[1].set_xticks(x)
    axs2[1].set_xticklabels(labels)
    axs2[1].set_title("2. Realized Revenue Stack: Bankable PPA + Spot Bonus (08h-20h)")
    axs2[1].set_ylabel("Annual Revenue (M€/yr)")
    axs2[1].legend()

    plt.tight_layout()
    p2 = plot_dir / "02_deterministic_cycling_and_revenues.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p2}")


def plot_stochastic_20saa_suite(
    df_stoch: pd.DataFrame,
    scen_dist_dict: dict,
    monthly_shapes_saa: dict,
    test_name: str
):
    """
    Plots for Script 2 (`stochastic_20_scenarios_saa`):
    True Two-Stage Stochastic Analysis across S=20 SAA scenarios (P50, P90, VaR90, CVaR90).
    """
    plot_dir = _get_plot_dir(test_name)
    ppa_df = df_stoch[df_stoch["Contract_Profile"] != "Merchant_Only"]
    merch_df = df_stoch[df_stoch["Contract_Profile"] == "Merchant_Only"]

    # --- PLOT 1: True P50, P90, VaR90, CVaR90 Risk Fan & Tail Protection ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Two-Stage Stochastic Model (S = 20 SAA Scenarios): P50, P90, VaR 90 & CVaR 90", fontsize=15, fontweight="bold")

    # 1. Risk Fan Chart on 08h-20h (P50 vs P90 vs VaR90 vs CVaR90)
    f_df = ppa_df[ppa_df["Contract_Profile"] == "08h-20h"]
    axs[0, 0].plot(f_df["Allowance_Pct"], f_df["Annualized_P50_Bankable_MEUR"], marker="o", linewidth=2.5, color="navy", label="P50 (Median across 20 scenarios)")
    axs[0, 0].plot(f_df["Allowance_Pct"], f_df["Annualized_P90_Bankable_MEUR"], marker="s", linewidth=2.2, color="royalblue", label="P90 (10th Percentile)")
    axs[0, 0].plot(f_df["Allowance_Pct"], f_df["Annualized_VaR90_Bankable_MEUR"], marker="^", linestyle="--", linewidth=2, color="darkorange", label="VaR 90 (Discrete 10% Quantile)")
    axs[0, 0].plot(f_df["Allowance_Pct"], f_df["Annualized_CVaR90_Bankable_MEUR"], marker="d", linestyle=":", linewidth=2.5, color="crimson", label="CVaR 90 (Worst 10% Tail Mean)")
    axs[0, 0].fill_between(f_df["Allowance_Pct"], f_df["Annualized_CVaR90_Bankable_MEUR"], f_df["Annualized_P50_Bankable_MEUR"], color="royalblue", alpha=0.15, label="P50 - CVaR90 Risk Spread")
    axs[0, 0].set_title("1. Stochastic Bankable PPA Revenue (08h-20h across 20 Scenarios)")
    axs[0, 0].set_xlabel("Free Allowance Quota (%)")
    axs[0, 0].set_ylabel("Annualized Bankable Revenue (M€/yr)")
    axs[0, 0].legend(fontsize=8.5)

    # 2. Full 20-Scenario Distribution Histogram (0% vs 10% Allowances on 08h-20h)
    dist_rows = []
    for label_k, d_map in scen_dist_dict.items():
        sorted_vals = np.sort(list(d_map.values()))
        for idx_s, val in enumerate(sorted_vals):
            dist_rows.append({"Configuration": label_k, "Rank": f"S{idx_s+1:02d}", "Bankable_Rev_MEUR": val})
    df_dist = pd.DataFrame(dist_rows)
    sns.barplot(data=df_dist, x="Rank", y="Bankable_Rev_MEUR", hue="Configuration", palette=["indianred", "mediumseagreen"], ax=axs[0, 1])
    axs[0, 1].axvspan(-0.5, 1.5, color="red", alpha=0.15, label="Worst 10% Tail (CVaR90 Zone: S01-S02)")
    axs[0, 1].set_title("2. Sorted Revenue Distribution Across the 20 SAA Scenarios (S01=Worst, S20=Best)")
    axs[0, 1].set_xlabel("Ordered SAA Scenarios (5% probability each)")
    axs[0, 1].set_ylabel("Net Bankable PPA Revenue (M€/yr)")
    axs[0, 1].tick_params(axis="x", rotation=45)
    axs[0, 1].legend(fontsize=8)

    # 3. Seasonal Stage-1 Baseload Commitment B_m (Here-and-Now across 20 scenarios)
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for label, s_dict in monthly_shapes_saa.items():
        axs[1, 0].plot(months, [s_dict[m] for m in months], marker="o", linewidth=2.2, label=label)
    axs[1, 0].set_xticks(months)
    axs[1, 0].set_xticklabels(month_labels)
    axs[1, 0].set_title("3. Stage-1 Robust Monthly Baseload Commitment B_m (MW)")
    axs[1, 0].set_ylabel("Here-and-Now Baseload B_m (MW)")
    axs[1, 0].legend(fontsize=8.5)

    # 4. Bankable IRR (at P90) vs Total IRR (at P90) + Merchant_Only Benchmark
    for prof, grp in ppa_df.groupby("Contract_Profile"):
        l1 = axs[1, 1].plot(grp["Allowance_Pct"], grp["IRR_Bankable_pct"], marker="o", linewidth=2.2, label=f"{prof} (Bankable P90 IRR)")
        axs[1, 1].plot(grp["Allowance_Pct"], grp["IRR_Total_pct"], marker="s", linestyle="--", color=l1[0].get_color(), linewidth=2.2, label=f"{prof} (Total P90 IRR + Spot)")
    if not merch_df.empty:
        m_irr = merch_df["IRR_Total_pct"].iloc[0]
        axs[1, 1].axhline(m_irr, color="darkorange", linestyle="-.", linewidth=2, label=f"Merchant_Only P90 IRR ({m_irr:.2f}%)")
    axs[1, 1].axhline(6.5, color="black", linestyle=":", label="WACC (6.5%)")
    axs[1, 1].set_title("4. Bankable P90 IRR vs Total P90 IRR Across Profiles")
    axs[1, 1].set_xlabel("Allowance Quota (%)")
    axs[1, 1].set_ylabel("P90 Project IRR (%)")
    axs[1, 1].legend(fontsize=8)

    plt.tight_layout()
    p1 = plot_dir / "01_stochastic_20saa_risk_and_distribution.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p1}")

    # --- PLOT 2: Penalties Avoided, Excess Shortfall & BESS Cycling Across 20 Scenarios ---
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig2.suptitle("Two-Stage Stochastic (20 SAA Scenarios): Penalty Protection & Battery Cycling", fontsize=15, fontweight="bold")

    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Annual_Avoided_Penalty_MEUR", hue="Contract_Profile", marker="o", linewidth=2.5, ax=axs2[0])
    axs2[0].set_title("1. Expected Spot Penalties Avoided by Allowances (M€/yr)")
    axs2[0].set_ylabel("Avoided Spot Penalty (M€/yr)")

    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Annual_Expected_Penalty_MEUR", hue="Contract_Profile", marker="s", linewidth=2.5, ax=axs2[1])
    axs2[1].set_title("2. Residual Excess Shortfall Penalty Paid (M€/yr)")
    axs2[1].set_ylabel("Residual Paid Penalty (M€/yr)")

    sns.lineplot(data=ppa_df, x="Allowance_Pct", y="Annual_BESS_Cycles", hue="Contract_Profile", marker="^", linewidth=2.5, ax=axs2[2])
    if not merch_df.empty:
        axs2[2].axhline(merch_df["Annual_BESS_Cycles"].iloc[0], color="darkorange", linestyle="--", label="Merchant_Only Cycles")
    axs2[2].set_title("3. Expected Annual Battery Cycling (EFC / yr)")
    axs2[2].set_ylabel("Equivalent Full Cycles / yr")
    axs2[2].legend()

    plt.tight_layout()
    p2 = plot_dir / "02_stochastic_20saa_penalties_and_cycling.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p2}")