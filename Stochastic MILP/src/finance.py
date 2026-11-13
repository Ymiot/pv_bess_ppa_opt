import numpy as np

def compute_crf(wacc: float, lifetime_years: int) -> float:
    """Computes the Capital Recovery Factor (CRF) for CAPEX annualization."""
    return (wacc * (1 + wacc) ** lifetime_years) / ((1 + wacc) ** lifetime_years - 1)

def _compute_npv_and_irr(annual_net_revenue_eur: float, total_capex: float, annual_opex: float, lifetime: int, wacc: float, tax_rate: float):
    """Internal helper to compute post-tax Cash Flows, NPV, and IRR."""
    annual_depreciation = total_capex / lifetime
    ebitda = annual_net_revenue_eur - annual_opex
    taxable_income = max(0.0, ebitda - annual_depreciation)
    taxes = taxable_income * tax_rate
    net_annual_cf = ebitda - taxes

    cash_flows = np.zeros(lifetime + 1)
    cash_flows[0] = -total_capex
    cash_flows[1:] = net_annual_cf

    years = np.arange(lifetime + 1)
    npv = float(np.sum(cash_flows / ((1 + wacc) ** years)))

    irr = np.nan
    if net_annual_cf > 0:
        roots = np.roots(cash_flows[::-1])
        real_roots = roots[np.isreal(roots)].real
        pos_roots = [r for r in real_roots if r > 0]
        if pos_roots:
            best_r = min(pos_roots, key=lambda r: abs(r - 1.0))
            irr = (1.0 / best_r) - 1.0

    return npv / 1e6, irr * 100.0, net_annual_cf / 1e6

def evaluate_project_finance(
    bankable_ppa_revenue_eur: float,
    merchant_bonus_revenue_eur: float,
    pv_mw: float,
    bess_power_mw: float,
    bess_energy_mwh: float,
    cfg: dict
) -> dict:
    """
    Evaluates Project Finance metrics under two views:
    1. Bankable View (Contracted PPA Baseload revenues ONLY - used by banks).
    2. Total View (Contracted PPA + Merchant Grid Bonus sales - used by equity investors).
    """
    c_cfg = cfg["capex_opex"]
    p_cfg = cfg["project"]
    
    lifetime = int(p_cfg["lifetime_years"])
    wacc = float(p_cfg["wacc_real"])
    tax_rate = float(p_cfg["tax_rate"])

    capex_pv = pv_mw * c_cfg["pv_capex_eur_per_mw"]
    capex_bess = (
        bess_power_mw * c_cfg["bess_power_capex_eur_per_mw"]
        + bess_energy_mwh * c_cfg["bess_energy_capex_eur_per_mwh"]
    )
    total_capex = capex_pv + capex_bess

    annual_opex = (
        pv_mw * c_cfg["pv_opex_eur_per_mw_yr"]
        + bess_power_mw * c_cfg["bess_opex_eur_per_mw_yr"]
    )

    # 1. Bankable Case (Baseload PPA Revenue Only)
    npv_bank, irr_bank, cf_bank = _compute_npv_and_irr(
        bankable_ppa_revenue_eur, total_capex, annual_opex, lifetime, wacc, tax_rate
    )

    # 2. Total Case (Baseload PPA + Merchant Grid Bonus)
    total_revenue_eur = bankable_ppa_revenue_eur + merchant_bonus_revenue_eur
    npv_tot, irr_tot, cf_tot = _compute_npv_and_irr(
        total_revenue_eur, total_capex, annual_opex, lifetime, wacc, tax_rate
    )

    return {
        "CAPEX_Total_MEUR": total_capex / 1e6,
        "CAPEX_BESS_MEUR": capex_bess / 1e6,
        "Bankable_PPA_Rev_MEUR": bankable_ppa_revenue_eur / 1e6,
        "Merchant_Bonus_Rev_MEUR": merchant_bonus_revenue_eur / 1e6,
        "Total_Rev_MEUR": total_revenue_eur / 1e6,
        "NPV_Bankable_MEUR": npv_bank,
        "IRR_Bankable_pct": irr_bank,
        "NPV_Total_MEUR": npv_tot,
        "IRR_Total_pct": irr_tot,
    }