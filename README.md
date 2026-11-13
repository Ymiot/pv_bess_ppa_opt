# Co-Located PV + BESS Optimization under Baseload PPAs with Non-Delivery Allowances

## 1. Project Overview & Research Strategy

This repository implements a quantitative optimization framework—spanning **Deterministic Perfect-Foresight MILP/LP** and **Two-Stage Stochastic Programming with $\text{CVaR}_{90}$**—to size, operate, and price hybrid Solar PV + Battery Energy Storage Systems (BESS) under shaped Baseload Power Purchase Agreements (PPAs) with **Non-Delivery Allowances** in the Danish electricity market (`DK1`).

### Repository Architecture
* **`data/config/`**: Contains `project_params.yml` with technical, financial (CAPEX, OPEX, WACC, tax rate), and contractual parameters.
* **`data/raw/`**: Cached historical hourly Day-Ahead Spot prices (Energinet `DK1`) and ERA5 solar PV generation profiles (`2020` and `2022`, normalized to 8,760 hours).
* **`src/`**: Core optimization and analytics engine:
  * `data_loader.py`: Historical API fetching and Sample Average Approximation (SAA) block-bootstrapping scenario generation.
  * `stochastic_ems.py`: Pyomo optimization engine (Deterministic and Two-Stage Stochastic MILP/LP formulations).
  * `finance.py`: Project Finance evaluation separating **Bankable Cash Flows** (PPA Baseload only) from **Total Equity Cash Flows** (PPA + Merchant Grid Bonus).
  * `plotting.py`: Automated generation of publication-ready figures.
* **`scripts/`**: Executable simulation pipelines (`run_single_test.py`, `run_deterministic_benchmark.py`, `run_stochastic_20_scenarios.py`).
* **`plots/`**: Output directory where each simulation saves its dedicated figures and CSV summary tables.

---

## 2. Simulation Roadmap & Modeling Stages

### Stage A: Synthetic Stress-Test & Contract Profile Exploration (`run_single_test.py`)
* **Setup**: Evaluates a 100 MWp PV plant with a **PV-only charged BESS** (no grid charging allowed, export-only connection) across contrasted synthetic weather/price regimes (Sunny Cannibalization, Calm 2020, Variable Spread, Cloudy Winter, and 2022 Dunkelflaute Crisis).
* **Contract Profiles Tested**:
  1. `Merchant_Only`: 0 MW Baseload benchmark (100% direct Spot market sales and arbitrage).
  2. `24/7`: Continuous round-the-clock baseload delivery.
  3. `08h-20h`: Daytime solar peakload window (12h/day).
  4. `12h-00h`: Afternoon and evening peak window (12h/day).
  5. `18h-20h`: Evening super-peak window (2h/day).
* **Decision Variables & Framework**: Fixed solar capacity (100 MWp) and fixed baseline PPA strike price (€75/MWh) across allowance quotas ranging from **0% to 15%** of active contract hours. The model co-optimizes:
  * Promised Baseload power ($B_{\text{baseload}}$ in MW),
  * BESS power capacity ($\text{MW}$),
  * BESS energy storage capacity ($\text{MWh}$).
* **Two-Tier Non-Delivery & Penalty Mechanism**:
  * **Tier 1 (Free Allowance Quota)**: Up to $X\%$ of active contract hours can be curtailed at **€0/MWh penalty** (and €0 revenue).
  * **Tier 2 (Penalized Excess Shortfall)**: Whenever the battery runs empty after the free allowance quota is exhausted, unexcused shortfall is penalized (configurable as `spot_price` = $\max(0, P^{\text{DA}}_t)$, `min_spot_ppa` = $\min(P^{\text{DA}}_t, P_{\text{PPA}})$, or `max_spot_ppa` = $\max(P^{\text{DA}}_t, P_{\text{PPA}})$).
* **Key Findings & Next Commercial Steps**:
  * Under a uniform €75/MWh PPA price, the **`08h-20h`** profile achieves the highest bankability because its delivery window aligns naturally with solar generation, whereas `12h-00h` and `18h-20h` require heavy evening storage.
  * However, in real power markets, an evening-peaking contract (`12h-00h` or `18h-20h`) commands a significantly higher market price than a daytime `08h-20h` profile.
  * **Upcoming Reverse-Pricing Extension (Iso-IRR / WACC Floor)**: Instead of fixing $P_{\text{PPA}} = 75\text{ €/MWh}$ and observing the resulting IRR, the next phase will fix the **target Bankable P90 IRR at the WACC floor (6.5%)** and solve for the **minimum required PPA strike price $P_{\text{PPA}}^*$** as a function of the contract window and the allowance percentage $X\%$. This will directly price both the time-of-day profile and the marginal value of each allowance percentage point.

---

### Stage B: 52-Week Deterministic Benchmark — 2020 vs. 2022 (`run_deterministic_benchmark.py`)
* **Why This Model Is Deterministic**: When the optimization engine is executed on a single historical year ($S = 1$), the two-stage model collapses into a **Deterministic Perfect-Foresight (Wait-and-See / Oracle)** benchmark. In this setting, statistical tail metrics ($\text{P50}, \text{P90}, \text{VaR}_{90}, \text{CVaR}_{90}$) are degenerate ($\text{P50} = \text{P90} = \text{CVaR}_{90}$) and are therefore omitted in favor of realized deterministic metrics.
* **Purpose**:
  1. Quantify how drastically optimal monthly baseload shaping ($B_1, \dots, B_{12}$) and BESS sizing shift between a **low-price solar cannibalization regime (2020)** and an **extreme energy crisis regime (2022)**.
  2. Perform **Out-of-Sample Cross-Validation (Crash-Testing)**:
     * *2020 Contract tested in 2022*: Evaluate the aggressive monthly baseload contract sized on 2020 when subjected to 2022 crisis prices (quantifying the multi-million-euro Spot penalty exposure and how Non-Delivery Allowances cushion the shock).
     * *2022 Contract tested in 2020*: Evaluate the conservative 2022 contract when operated during a calm 2020 year (quantifying the opportunity cost of over-conservatism).

---

### Stage C: True Two-Stage Stochastic Model on 20 SAA Scenarios (`run_stochastic_20_scenarios.py`)
* **Scenario Generation ($S = 20$)**: Generates 20 equiprobable 8,760-hour scenarios ($\pi_\omega = 5\%$ each) via **7-day Circular Block-Bootstrapping** and regime blending between 2020 and 2022, preserving diurnal solar-price negative correlation and multi-day *Dunkelflaute* cloud persistence.
* **Statistical Validity of Risk Metrics**: With $S = 20$ scenarios, the revenue distribution is non-degenerate, making **$\text{P50}$** (median), **$\text{P90} / \text{VaR}_{90}$** (10th percentile / 2nd worst scenario), and **$\text{CVaR}_{90}$** (expected revenue in the worst 10% tail) mathematically rigorous and directly usable for project finance debt sizing.

---

## 3. Mathematical & Algorithmic Explanation of the Two-Stage Stochastic Solver (`solve_two_stage_stochastic_ems`)

### 3.1. Decision Hierarchy: Stage 1 ("Here-and-Now") vs. Stage 2 ("Wait-and-See")

The algorithm models the exact chronological information asymmetry faced by a renewable IPP signing a PPA prior to commercial operation:

1. **Stage 1 — *Here-and-Now* Decisions (Scenario-Independent)**:
   Taken **before** the weather and Spot price realization $\omega \in \Omega = \{1, \dots, S\}$ is known. These variables are identical across all 20 scenarios:
   * $B_{\text{baseload}} \ge 0$: The firm Baseload power (MW) promised to the offtaker during active contract hours $\mathcal{T}_{\text{active}}$.
   * $P^{\text{BESS}}_{\max}, E^{\text{BESS}}_{\max} \ge 0$: Installed BESS power capacity (MW) and storage capacity (MWh).
   * $\zeta \in \mathbb{R}$: The auxiliary Value-at-Risk ($\text{VaR}_\alpha$) threshold variable used for tail-risk optimization.

2. **Stage 2 — *Wait-and-See* Recourse Decisions (Scenario-Dependent, indexed by $\omega, t$)**:
   Once scenario $\omega$ unfolds, the Energy Management System (EMS) dispatches the plant hour by hour ($t \in \mathcal{T}$) in response to available solar power $P^{\text{PV}}_{\omega,t}$ and Day-Ahead Spot prices $P^{\text{DA}}_{\omega,t}$:
   * Physical PV & BESS flows: $p^{\text{pv}\to\text{grid}}_{\omega,t}$, $p^{\text{pv}\to\text{bess}}_{\omega,t}$, $p^{\text{bess}\to\text{grid}}_{\omega,t}$, and State of Charge $\text{SoC}_{\omega,t}$.
   * Commercial power split at the Point of Interconnection (POI):
     $$p^{\text{pv}\to\text{grid}}_{\omega,t} + p^{\text{bess}\to\text{grid}}_{\omega,t} = q^{\text{ppa\_deliv}}_{\omega,t} + q^{\text{merchant\_bonus}}_{\omega,t}$$

---

### 3.2. The Three-Bucket Contractual Balance During Active Hours

At every active contract hour $t \in \mathcal{T}_{\text{active}}$ in scenario $\omega$, the Stage-1 commitment $B_{\text{baseload}}$ must be accounted for across three mutually complementary Stage-2 buckets:

$$B_{\text{baseload}} = \underbrace{q^{\text{ppa\_deliv}}_{\omega,t}}_{\substack{\text{1. Physical Delivery}\\\text{Earns } +P_{\text{PPA}}\text{ (€/MWh)}}} + \underbrace{q^{\text{allow\_free}}_{\omega,t}}_{\substack{\text{2. Free Allowance Used}\\\text{€0 Revenue, €0 Penalty}}} + \underbrace{q^{\text{excess\_shortfall}}_{\omega,t}}_{\substack{\text{3. Unexcused Default (Empty BESS)}\\\text{Penalized at } -\lambda^{\text{pen}}_{\omega,t}\text{ (€/MWh)}}}$$

* **Priority 1 (Physical Delivery)**: As long as solar generation or stored BESS energy is available, the solver routes power into $q^{\text{ppa\_deliv}}_{\omega,t}$ to earn the fixed PPA price $P_{\text{PPA}}$.
* **Priority 2 (Strategic Allocation of Free Allowances)**: During multi-day cloudy periods (*Dunkelflaute*) when BESS energy is scarce, the solver excuses non-delivery via $q^{\text{allow\_free}}_{\omega,t}$, bounded at each hour by $0 \le q^{\text{allow\_free}}_{\omega,t} \le B_{\text{baseload}}$ and capped over the period by the contractual allowance budget $N_{\text{allow}} = \lfloor X\% \cdot |\mathcal{T}_{\text{active}}| \rfloor$:
  $$\sum_{t \in \mathcal{T}_{\text{active}}} q^{\text{allow\_free}}_{\omega,t} \le N_{\text{allow}} \cdot B_{\text{baseload}} \quad \text{(Exact LP Energy-Allowance Bank)}$$
  *(or via binary indicators $u_{\omega,t} \in \{0,1\}$ with $\sum_t u_{\omega,t} \le N_{\text{allow}}$ in strict MILP mode).*
  Because excess shortfall is penalized at a Spot-linked rate $\lambda^{\text{pen}}_{\omega,t}$, the solver automatically reserves its finite battery energy and triggers $q^{\text{allow\_free}}_{\omega,t}$ during the hours with the **highest Spot price spikes** ($400\text{–}550\text{ €/MWh}$).
* **Priority 3 (Penalized Excess Shortfall)**: If the Stage-1 commitment $B_{\text{baseload}}$ is too aggressive, the battery empties and the free allowance quota is exhausted; the residual deficit spills into $q^{\text{excess\_shortfall}}_{\omega,t} > 0$, incurring severe financial penalties.

---

### 3.3. Linearized $\text{CVaR}_{90}$ Coupling (Rockafellar-Uryasev Formulation)

For each scenario $\omega \in \{1, \dots, S\}$, net annualized **Bankable PPA Profit** $\Pi^{\text{Bank}}_{\omega}$ is defined strictly on contracted PPA cash flows (excluding merchant upside, as required by project finance lenders):

$$\Pi^{\text{Bank}}_{\omega} = \sum_{t \in \mathcal{T}} \left( P_{\text{PPA}} \, q^{\text{ppa\_deliv}}_{\omega,t} - \lambda^{\text{pen}}_{\omega,t} \, q^{\text{excess\_shortfall}}_{\omega,t} - c_{\text{deg}} \, p^{\text{bess}\to\text{grid}}_{\omega,t} \right) - \text{CAPEX}^{\text{annualized}}_{\text{BESS}}$$

while **Merchant Bonus Revenue** is tracked separately:
$$\Pi^{\text{Merch}}_{\omega} = \sum_{t \in \mathcal{T}} \max(0, P^{\text{DA}}_{\omega,t}) \, q^{\text{merchant\_bonus}}_{\omega,t}$$

To maximize the P90 / tail-risk bankability without non-linear sorting inside the solver, the model applies the **Rockafellar & Uryasev (2000)** linear reformulation at confidence level $\alpha = 0.90$ using shortfall variables $z_\omega \ge 0$:

$$z_\omega \ge \zeta - \Pi^{\text{Bank}}_{\omega}, \quad z_\omega \ge 0 \quad \forall \omega \in \Omega$$

$$\text{CVaR}_{90}\!\left(\Pi^{\text{Bank}}\right) = \zeta - \frac{1}{1 - \alpha} \sum_{\omega=1}^{S} \pi_\omega \, z_\omega$$

At optimality, $\zeta^*$ equals the **$\text{VaR}_{90}$ ($\text{P90}$)** revenue threshold, and $\text{CVaR}_{90}$ equals the average revenue of the worst $10\%$ scenarios.

---

### 3.4. How the Solver Finds the Optimum: Extensive Form vs. Iterative Feedback Loop

Conceptually, the optimization acts as a **closed-loop feedback system** between Stage 1 and Stage 2:
1. If Stage 1 proposes a candidate $\left(B_{\text{baseload}}, P^{\text{BESS}}_{\max}, E^{\text{BESS}}_{\max}\right)$, the 20 scenarios decouple into 20 independent deterministic dispatch problems yielding $\left(\Pi^{\text{Bank}}_1, \dots, \Pi^{\text{Bank}}_{20}\right)$.
2. If $B_{\text{baseload}}$ is pushed too high to capture revenue in sunny scenarios, the worst crisis scenarios ($\omega \in \text{Tail}_{10\%}$) run out of battery and free allowances, causing $q^{\text{excess\_shortfall}}_{\omega,t}$ and $z_\omega$ to spike, which immediately penalizes $\text{CVaR}_{90}$.

Computationally, rather than running a slow external loop, `solve_two_stage_stochastic_ems` assembles the **Block-Angular Extensive Form (Deterministic Equivalent)** in Pyomo and solves all 20 scenarios and Stage-1 variables **simultaneously in a single matrix pass** maximizing:

$$\max_{B, \text{BESS}, \zeta, \mathbf{y}_\omega} \quad (1 - \lambda) \, \mathbb{E}_\omega\!\left[\Pi^{\text{Bank}}_\omega\right] \;+\; \lambda \, \text{CVaR}_{90}\!\left(\Pi^{\text{Bank}}_\omega\right) \;+\; w_{\text{merch}} \, \mathbb{E}_\omega\!\left[\Pi^{\text{Merch}}_\omega\right]$$

where $\lambda = 0.60$ weights bankable tail protection ($\text{CVaR}_{90}$) against expected PPA revenue ($\mathbb{E}[\Pi^{\text{Bank}}]$), and $w_{\text{merch}} = 0.15$ ensures excess solar energy is monetized on the Spot market whenever the PPA commitment is already satisfied.