# Quantum Bayesian Networks for Cyber-Physical Threat Detection in Pipeline SCADA Systems

This repository contains the full implementation, CPT derivation, and inference experiments for the paper:

> **Quantum Bayesian Networks for Cyber-Physical Threat Prediction in Pipeline SCADA Systems**
> David Alabintei — University of Maryland, Civil & Environmental Engineering

The work applies Quantum Bayesian Networks (QBNs) to real-time cyber-physical threat detection in a SCADA water distribution pipeline system, using the Hardware-in-the-Loop (HIL) Water Distribution Testbed dataset (Faramondi et al., *IEEE Access*, 2021).

---

## Overview

A two-subsystem Bayesian network models attack propagation across a SCADA pipeline:

```
AttackType_1 ──→ AttackType_2
     │                │
     ▼                ▼
ValveState_1    ValveState_2
PumpState_1     FlowState_2
TankState_1     TankState_2
```

- **S1** (real subsystem) — controlled by PLC1: tanks 1–5, pumps 1–4, valves 1–19
- **S2** (simulated subsystem) — controlled by PLC2: tank 6, valve 20, flow sensor 1
- **Attack states**: Normal, MiTM, DoS, Physical

Seven inference methods are compared across four diagnostic queries, from exact inference through classical and quantum sampling to variational inference and sensitivity analysis.

---

## Dataset

**HIL Water Distribution Testbed** — Faramondi, L. et al. "A Hardware-in-the-Loop Water Distribution Testbed Dataset for Cyber-Physical Security Testing." *IEEE Access* (2021).

The dataset is **not included** in this repository. Download it separately and place it at:
```
../cyber pipeline dataset/
    Physical dataset/phy_att_1.csv  ...  phy_att_4.csv
    README.xlsx
```
Update the path constants at the top of `cyber_bn_cpt_generation.ipynb` to match your local directory.

---

## Repository Structure

```
.
├── cyber_bn_cpt_generation.ipynb      # CPT derivation (Sets 1, 2, 3) from HIL dataset
├── cyber_inference_experiments.ipynb  # Main experiment notebook (Phases 1–5)
├── cyber_qubayes.ipynb                # Exploratory QBN queries (supplementary)
├── requirements.txt                   # Python dependencies
├── phase5_sensitivity.pdf             # Generated sensitivity analysis figure
│
└── qubayes/                           # QBN framework (Krebs et al., ICMCI 2024)
    └── qubayes/
        ├── qubayes_tools.py               # Core QBN: Graph, Node, QBNQuery
        ├── cyber_attack_bn.py             # CPT constants — single source of truth
        ├── variational_inference_cyber.py # Born machines, COBYLA optimiser
        ├── variational_inference.py       # General VI framework
        └── tests/                         # Unit tests
```

---

## Installation

```bash
# Create and activate environment
conda create -n qubayes python=3.11
conda activate qubayes

# Install dependencies
pip install -r requirements.txt

# Install the qubayes package in editable mode
pip install -e qubayes/
```

> **Note:** `qiskit==1.0.0` and `qiskit-aer>=0.13` are required. Qiskit 2.x is not yet supported. Pin your environment to avoid silent breakage.

---

## Reproducing Results

Run the notebooks in this order:

### Step 1 — Generate CPTs
Open and run **`cyber_bn_cpt_generation.ipynb`** top to bottom.

This notebook derives all three CPT sets from the HIL dataset:
- **Set 1** `P(AttackType_1)` — data-derived from attack window timestamps
- **Set 2** `P(AttackType_2 | AttackType_1)` — data-derived transition matrix with corrected PLC-targeting labelling
- **Set 3** `P(Sensor=Abnormal | AttackType)` — **hybrid methodology** (see below)

Final constants are printed at the end for copy-paste into `qubayes/qubayes/cyber_attack_bn.py`.

### Step 2 — Run inference experiments
Open and run **`cyber_inference_experiments.ipynb`** top to bottom.

| Phase | Content |
|-------|---------|
| 1a | Exact inference (pgmpy Variable Elimination) — ground truth |
| 1b | Classical Rejection Sampling (CRS) vs Quantum Rejection Sampling (QRS) |
| 1c | Optimal Grover iterations k — formula vs empirical sweep |
| 2  | Classical Likelihood Weighting (CLW) |
| 3  | Quantum Likelihood Weighting (QLW) |
| 4a | Classical Variational Inference (CAVI) |
| 4b | Quantum Variational Inference (QVI-RA0, QVI-RA1, QVI-BN) |
| 5  | Sensitivity analysis (OAT + Monte Carlo uncertainty bands) |
| 6  | IBM Quantum hardware execution — QRS and QVI on ibm_fez (Heron r2) |

---

## CPT Methodology — Set 3 (Hybrid)

Set 3 emission probabilities `P(Sensor=Abnormal | AttackType)` use a **hybrid methodology** — each column is derived by a different means appropriate to the attack mechanism:

| Column | Method | Value | Basis |
|--------|--------|-------|-------|
| Normal | ISA-TR84 false-alert floor | 0.01 | Real deployments have calibration drift; testbed shows zero false positives (artefact) |
| MiTM | Rule-based — stealth property | 0.10 | MITRE ATT&CK for ICS T0856/T0830: attacker replays legitimate values to SCADA |
| DoS | Rule-based — PLC starvation | 0.80 | MITRE ATT&CK for ICS T0814: uncontrolled process drifts, sensor readings become abnormal |
| Physical | Sensor-type tier system | tank=0.80, valve=0.60, pump/flow=0.55 | Functional role in water distribution; data anchor: S1 pump 290/550 ≈ 0.527 (Laplace α=0.5) |

Physical column tier values are applied **consistently across both subsystems** by sensor type — not by the specific fault scenarios in the dataset — because any subsystem can experience any fault type in a real deployment.

---

## Key Results

### Inference accuracy and efficiency (10 runs, N=400/512 shots — simulator)

| Method | Mean \|Err\| (avg Q1–Q4) | Notes |
|--------|--------------------------|-------|
| Exact (pgmpy VE) | 0 | Ground truth |
| **QRS** | **0.0028** | 99.7% acceptance rate; **123× speedup** over CRS for rare-evidence queries |
| CRS | 0.0058 | 0.81% acceptance for Q1/Q2 (rare evidence) |
| QVI-RA1 | 0.0109 | Best single-query result: Q4 error = 0.0034 |
| QVI-RA0 | 0.0146 | Simplest Born machine (2 params); best Q1 result = 0.0052 |
| CLW | 0.0077 | ESS 2.5% for Q1/Q2 — weight degeneracy |
| QLW | 0.0147 | No advantage over CLW; quantum circuit does not resolve weight degeneracy |
| CAVI | 0.143 | Mean-field collapse on Q1/Q2 (error ≈ 0.28); exact on Q3/Q4 (direct evidence) |

### Quantum advantage (Phase 1b — simulator)

| Query | Evidence rarity | QRS speedup | QRS std reduction |
|-------|----------------|-------------|-------------------|
| Q1/Q2 | P(e) = 0.0091 | **123×** | **6×** |
| Q3/Q4 | P(e) = 0.0319 | **29×** | **4×** |

### IBM Quantum hardware results (Phase 6 — ibm_fez, Heron r2)

QRS and QVI were evaluated on IBM `ibm_fez` for Q3/Q4 (evidence: A2=MiTM, k=4 Grover iterations, 4096 shots).

| Method | Q3 HW | Q3 \|Err\| | Q4 HW | Q4 \|Err\| | Circuit | 2Q gates |
|--------|-------|-----------|-------|-----------|---------|----------|
| QRS | 0.3185 | **0.3147** | 0.2448 | **0.1178** | 10q, d=102 | hundreds |
| QVI-RA0 | 0.6033 | 0.0299 | 0.3889 | 0.0263 | 2q, d=4 | 0 |
| QVI-RA1 | 0.6443 | **0.0111** | 0.3411 | 0.0216 | 2q, d=8 | 1 |
| QVI-BN | 0.6455 | 0.0123 | 0.3459 | **0.0167** | 2q, d=13 | 1 |

**Key finding:** QRS posteriors collapse toward the uniform prior (0.25) on hardware — the 10-qubit Grover circuit (depth 102 + amplification) exceeds the device's coherence budget. QVI Born machine circuits (2 qubits, depth 4–13) incur near-zero noise penalty, with hardware errors 10–30× smaller than QRS. This confirms that **circuit depth, not algorithm class, determines NISQ viability** for quantum probabilistic inference.

### Sensitivity (Phase 5)

- OAT: max \|Δposterior\| = 0.0076 for ±30% single-sensor perturbation — smaller than QRS shot noise
- MC (N=500, σ=0.05): Q1 P5 = 0.420 > uniform prior (0.25); Q2 P5 = 0.653 — qualitative conclusions robust across full uncertainty band
- Q3/Q4: perfectly invariant to sensor CPT perturbation (direct attack-state evidence bypasses sensor nodes)

---

## Queries

| ID | Query | Exact |
|----|-------|-------|
| Q1 | P(A2=Physical \| Flow2=Abn, Valve2=Abn) | 0.6658 |
| Q2 | P(A1=Normal \| Flow2=Abn, Valve2=Abn) | 0.6960 |
| Q3 | P(A1=Normal \| A2=MiTM) | 0.6332 |
| Q4 | P(A1=MiTM \| A2=MiTM) | 0.3626 |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26, <2.0 | Numerical computation |
| pandas | ≥2.0 | Data handling |
| openpyxl | ≥3.1 | Reading README.xlsx |
| pgmpy | ≥0.1.25 | Exact inference (Variable Elimination) |
| qiskit | ≥2.2 | Quantum circuit construction |
| qiskit-aer | ≥0.15 | Quantum simulation backend |
| qiskit-algorithms | ≥0.4 | Grover operator utilities |
| qiskit-ibm-runtime | ≥0.46 | IBM Quantum hardware execution (Phase 6) |
| tensorflow | ≥2.13, <2.16 | Variational inference (Born machine training) |
| matplotlib | ≥3.7 | Figures |

---

## QBN Framework

The `qubayes` package is based on:

> Krebs, F., Fuerntratt, H., Unterberger, R., & Graf, F. "Modeling Musical Knowledge with Quantum Bayesian Networks." *Proceedings of the International Conference on Content-based Multimedia Indexing* (2024).

Original repository: [github.com/JR-DIGITAL/qubayes](https://github.com/JR-DIGITAL/qubayes)

This work extends the framework with:
- A cyber-physical SCADA BN model (`cyber_attack_bn.py`)
- Quantum Likelihood Weighting (`QLW`) implementation
- Quantum Variational Inference with multiple ansatz variants (`variational_inference_cyber.py`)
- Full experimental comparison pipeline (`cyber_inference_experiments.ipynb`)
- IBM Quantum hardware execution and evaluation (Phase 6)

---

## References

- Faramondi, L. et al. "A Hardware-in-the-Loop Water Distribution Testbed Dataset for Cyber-Physical Security Testing." *IEEE Access* (2021).
- Krebs, F. et al. "Modeling Musical Knowledge with Quantum Bayesian Networks." *ICMCI* (2024).
- Low, G. H., Yoder, T. J., & Chuang, I. L. "Quantum inference on Bayesian networks." *Physical Review A* 89.6 (2014).
- MITRE ATT&CK for ICS — T0814 (Denial of Service), T0830 (Manipulate I/O Image), T0856 (Spoof Reporting Message).
- ISA-TR84.00.09 — Guidance for Safety Instrumented Systems (false-alert rates for process instrumentation).
- IEC 61511 — Functional Safety: Safety Instrumented Systems (diagnostic coverage ranges).

---

## Contact

David Alabintei — alabinte@umd.edu
University of Maryland, Department of Civil & Environmental Engineering
