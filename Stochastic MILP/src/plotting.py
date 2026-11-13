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
    monthly_shapes_records: list,
    df_cross: pd.DataFrame,
    test_name: str
):
    """
    Generates 4 publication-grade figure suites for the Deterministic Benchmark (2020 vs 2022):
    1. Monthly Baseload profiles (Jan-Dec) for all profiles (24/7, 08h-20h, 12h-20h) & allowances (0, 5, 10, 15%).
    2. Bankable IRR vs Total IRR (+ Merchant Bonus) and Merchant_Only benchmark.
    3. Co-optimized BESS Power (MW), Storage Capacity (MWh), and Annual Cycling.
    4. Full Cross-Scenario Out-of-Sample Matrix (2020 models tested on 2022 & 2022 models tested on 2020).
    """
    plot_dir = _get_plot_dir(test_name)
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ppa_profiles = ["24/7", "08h-20h", "12h-20h"]
    regimes = ["2020_Cannibalization", "2022_Energy_Crisis"]
    df_shapes = pd.DataFrame(monthly_shapes_records)

    # =========================================================================
    # PLOT 1: 2x3 Grid of Monthly Baseload Profiles (Jan-Dec) for All Models
    # =========================================================================
    fig1, axs1 = plt.subplots(2, 3, figsize=(19, 10), sharey=True)
    fig1.suptitle(
        "Optimal Monthly Baseload Profiles B_m (Jan - Dec) Across Contract Types & Allowances (0%, 5%, 10%, 15%)",
        fontsize=15, fontweight="bold"
    )
    palette_allow = {0.0: "#1f77b4", 5.0: "#2ca02c", 10.0: "#ff7f0e", 15.0: "#d62728"}

    for r_idx, reg in enumerate(regimes):
        for c_idx, prof in enumerate(ppa_profiles):
            ax = axs1[r_idx, c_idx]
            sub_s = df_shapes[(df_shapes["Year_Regime"] == reg) & (df_shapes["Contract_Profile"] == prof)]
            for allow_val, grp in sub_s.groupby("Allowance_Pct"):
                ax.plot(
                    grp["Month"], grp["Baseload_MW"],
                    marker="o", linewidth=2.2, color=palette_allow.get(allow_val, "black"),
                    label=f"Allow {int(allow_val)}%"
                )
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, rotation=30)
            ax.set_title(f"{reg} | Profile: {prof}", fontsize=12, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel("Monthly Promised Baseload (MW)")
            ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    p1 = plot_dir / "01_monthly_baseload_profiles_all_models.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p1}")

    # =========================================================================
    # PLOT 2: Bankable IRR vs Total IRR (+ Bonus) & Merchant_Only Comparison
    # =========================================================================
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig2.suptitle(
        "Deterministic IRR Analysis: Bankable IRR (Solid) vs Total IRR with Grid Bonus (Dashed) vs Merchant Only",
        fontsize=14, fontweight="bold"
    )
    for r_idx, reg in enumerate(regimes):
        ax = axs2[r_idx]
        sub_r = df_det[df_det["Year_Regime"] == reg]
        ppa_r = sub_r[sub_r["Contract_Profile"] != "Merchant_Only"]
        merch_r = sub_r[sub_r["Contract_Profile"] == "Merchant_Only"]

        for prof, grp in ppa_r.groupby("Contract_Profile"):
            l1 = ax.plot(grp["Allowance_Pct"], grp["IRR_Bankable_pct"], marker="o", linewidth=2.3, label=f"{prof} - Bankable IRR (PPA Only)")
            ax.plot(
                grp["Allowance_Pct"], grp["IRR_Total_pct"],
                marker="s", linestyle="--", linewidth=2.3, color=l1[0].get_color(), alpha=0.8,
                label=f"{prof} - Total IRR (+ Spot Bonus)"
            )
        if not merch_r.empty:
            m_irr = merch_r["IRR_Total_pct"].iloc[0]
            ax.axhline(m_irr, color="crimson", linestyle="-.", linewidth=2.5, label=f"Merchant_Only IRR ({m_irr:.2f}%)")
        ax.axhline(6.5, color="black", linestyle=":", linewidth=1.8, label="WACC Target (6.5%)")
        ax.set_title(f"IRR Performance in {reg}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Free Allowance Quota (%)")
        if r_idx == 0:
            ax.set_ylabel("Project IRR (%)")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    p2 = plot_dir / "02_irr_bankable_vs_total_and_merchant.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p2}")

    # =========================================================================
    # PLOT 3: Co-Optimized BESS Sizing (Power MW, Energy MWh) & Cycling
    # =========================================================================
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig3.suptitle("Co-Optimized BESS Sizing (Storage MWh, Power MW) & Annual Cycling (2020 vs 2022)", fontsize=15, fontweight="bold")
    ppa_all = df_det[df_det["Contract_Profile"] != "Merchant_Only"]

    sns.lineplot(data=ppa_all, x="Allowance_Pct", y="Optimal_BESS_Energy_MWh", hue="Contract_Profile", style="Year_Regime", markers=True, linewidth=2.4, ax=axs3[0])
    axs3[0].set_title("1. Optimal BESS Storage Capacity (MWh)")
    axs3[0].set_ylabel("BESS Energy (MWh)")

    sns.lineplot(data=ppa_all, x="Allowance_Pct", y="Optimal_BESS_Power_MW", hue="Contract_Profile", style="Year_Regime", markers=True, linewidth=2.4, ax=axs3[1])
    axs3[1].set_title("2. Optimal BESS Power Capacity (MW)")
    axs3[1].set_ylabel("BESS Power (MW)")

    sns.lineplot(data=ppa_all, x="Allowance_Pct", y="Annual_BESS_Cycles", hue="Contract_Profile", style="Year_Regime", markers=True, linewidth=2.4, ax=axs3[2])
    axs3[2].set_title("3. Annual Battery Cycling (Equivalent Full Cycles/yr)")
    axs3[2].set_ylabel("Cycles / yr")

    plt.tight_layout()
    p3 = plot_dir / "03_optimal_bess_sizing_and_cycling.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p3}")

    # =========================================================================
    # PLOT 4: Cross-Scenario Out-of-Sample Matrix (2020->2022 and 2022->2020)
    # =========================================================================
    fig4, axs4 = plt.subplots(2, 2, figsize=(17, 11))
    fig4.suptitle(
        "Out-of-Sample Cross-Evaluation: 2020 Designs Tested in 2022 Crisis vs 2022 Designs Tested in 2020",
        fontsize=15, fontweight="bold"
    )

    c_20_in_22 = df_cross[(df_cross["Evaluation_Direction"] == "Trained_2020_Tested_in_2022") & (df_cross["Contract_Profile"] != "Merchant_Only")]
    c_22_in_20 = df_cross[(df_cross["Evaluation_Direction"] == "Trained_2022_Tested_in_2020") & (df_cross["Contract_Profile"] != "Merchant_Only")]

    # Panel A: 2020 Model tested in 2022 - Bankable Revenue (In-Sample Oracle vs Out-of-Sample)
    sns.barplot(data=c_20_in_22, x="Allowance_Pct", y="OOS_Bankable_PPA_Rev_MEUR", hue="Contract_Profile", palette="Reds_d", ax=axs4[0, 0])
    axs4[0, 0].axhline(0, color="black", linewidth=1)
    axs4[0, 0].set_title("1. [2020 Model -> Tested in 2022] Realized Net PPA Revenue (M€/yr)")
    axs4[0, 0].set_ylabel("Net PPA Revenue in 2022 (M€/yr)")

    # Panel B: 2020 Model tested in 2022 - Total IRR vs Native 2022 Oracle IRR
    for prof, grp in c_20_in_22.groupby("Contract_Profile"):
        l = axs4[0, 1].plot(grp["Allowance_Pct"], grp["OOS_IRR_Total_pct"], marker="o", linewidth=2.3, label=f"{prof} (2020 Design in 2022)")
        axs4[0, 1].plot(grp["Allowance_Pct"], grp["Native_Oracle_IRR_Total_pct"], marker="s", linestyle="--", color=l[0].get_color(), alpha=0.75, label=f"{prof} (2022 Native Oracle)")
    axs4[0, 1].axhline(6.5, color="black", linestyle=":", label="WACC (6.5%)")
    axs4[0, 1].set_title("2. [2020 Model -> Tested in 2022] Total IRR vs 2022 Oracle IRR (%)")
    axs4[0, 1].set_ylabel("Total IRR in 2022 (%)")
    axs4[0, 1].legend(fontsize=8)

    # Panel C: 2022 Model tested in 2020 - Realized Net PPA Revenue
    sns.barplot(data=c_22_in_20, x="Allowance_Pct", y="OOS_Bankable_PPA_Rev_MEUR", hue="Contract_Profile", palette="Blues_d", ax=axs4[1, 0])
    axs4[1, 0].axhline(0, color="black", linewidth=1)
    axs4[1, 0].set_title("3. [2022 Model -> Tested in 2020] Realized Net PPA Revenue (M€/yr)")
    axs4[1, 0].set_ylabel("Net PPA Revenue in 2020 (M€/yr)")

    # Panel D: 2022 Model tested in 2020 - Total IRR vs Native 2020 Oracle IRR
    for prof, grp in c_22_in_20.groupby("Contract_Profile"):
        l = axs4[1, 1].plot(grp["Allowance_Pct"], grp["OOS_IRR_Total_pct"], marker="o", linewidth=2.3, label=f"{prof} (2022 Design in 2020)")
        axs4[1, 1].plot(grp["Allowance_Pct"], grp["Native_Oracle_IRR_Total_pct"], marker="s", linestyle="--", color=l[0].get_color(), alpha=0.75, label=f"{prof} (2020 Native Oracle)")
    axs4[1, 1].axhline(6.5, color="black", linestyle=":", label="WACC (6.5%)")
    axs4[1, 1].set_title("4. [2022 Model -> Tested in 2020] Total IRR vs 2020 Oracle IRR (%)")
    axs4[1, 1].set_ylabel("Total IRR in 2020 (%)")
    axs4[1, 1].legend(fontsize=8)

    plt.tight_layout()
    p4 = plot_dir / "04_cross_scenario_out_of_sample_matrix.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p4}")