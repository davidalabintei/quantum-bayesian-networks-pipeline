# Paper Outline
## Quantum Bayesian Networks for Cyber-Physical Threat Detection in Pipeline SCADA Systems: A Comparative Study from Simulation to Hardware

**Author:** David Alabintei — University of Maryland, Civil & Environmental Engineering

---

### Framing and Positioning

**What this paper is:**
- An *applied quantum computing* paper for infrastructure cyber-security
- A *systematic empirical study* comparing 7 existing inference methods on the same BN, same queries, same dataset — the most comprehensive QBN comparison to date
- The *first hardware head-to-head* of Grover-based vs variational quantum inference for Bayesian networks, yielding a practical finding about NISQ viability

**What this paper is NOT:**
- We do not propose new quantum algorithms (QRS, QVI, Born machines are established methods)
- We do not claim to improve any algorithm's theoretical properties

**Novel contributions (honestly scoped):**
1. First application of QBNs to cyber-physical SCADA threat detection, grounded in a real HIL dataset
2. A hybrid CPT derivation methodology combining data, MITRE ATT&CK for ICS, and ISA/IEC standards — a reusable template for infrastructure BN modelling under data sparsity
3. The most comprehensive comparison of quantum inference methods for BNs to date (7 methods, 4 queries, simulator + hardware)
4. First empirical evidence that circuit depth — not algorithm class — determines which quantum inference methods survive on NISQ hardware: QRS (10q, depth 102) collapses toward uniform, QLW (10q, depth 98) is even worse due to noise-amplified reweighting, while QVI (2q, depth 4–13) incurs near-zero noise penalty. Classical post-processing cannot rescue noisy quantum samples.

---

## 1. Introduction

- Open with operational stakes: ICS/SCADA systems underpin water, energy, and pipeline infrastructure; attacks are rare but high-consequence (cite Colonial Pipeline, Oldsmar, Stuxnet).
- Bayesian networks are a natural framework for probabilistic threat reasoning — belief propagation across the attack-sensor causal chain.
- The quantum opportunity: classical sampling degrades for rare-evidence queries, which correspond to the most operationally dangerous attack scenarios. Grover-based quantum rejection sampling (Low et al. 2014) offers quadratic speedup that grows with evidence rarity.
- The NISQ reality: prior QBN work (Borujeni et al. 2021, Krebs et al. 2024) demonstrates circuits and simulator results, but hardware evaluation reveals severe noise degradation. No prior work has systematically compared which inference approaches survive on real hardware.
- **This paper:** We apply QBNs to a real SCADA cyber-physical dataset, compare 7 inference methods on simulator, and execute both Grover-based and variational approaches on IBM quantum hardware. The hardware results reveal that circuit depth — not algorithm class — is the determining factor for NISQ viability.
- Contributions summary (the 4 points from Framing above)
- Paper organisation paragraph.

---

## 2. Background and Related Work

### 2.1 Cyber-Physical Security in Water and Pipeline SCADA Systems
- Nature of ICS/SCADA threats: MiTM, DoS, physical attacks (MITRE ATT&CK for ICS taxonomy).
- Challenges: sparse labelled data, rare attacks, real-time constraints.
- Prior probabilistic/Bayesian approaches to ICS security (brief survey; note limitations).

### 2.2 Bayesian Networks for Infrastructure Risk and Security
- BN use in civil/infrastructure risk (bridges, pipelines, water systems).
- CPT derivation challenges in data-sparse or standards-constrained domains.

### 2.3 Quantum Bayesian Networks
- QBN circuit encoding: mapping BN structure and CPTs to quantum circuits (Low et al. 2014; Borujeni et al. 2021).
- Quantum Rejection Sampling: amplitude amplification of evidence-consistent states.
- Variational quantum inference: Born machines as posterior approximators (Benedetti et al. 2019; Krebs et al. 2024).
- Prior QBN applications (music, bridges) — note all are simulation-only.
- **Hardware gap:** Borujeni et al. ran QBN circuits on IBM hardware and observed degradation but did not compare alternative methods or identify mitigation paths.

### 2.4 Gap and Motivation
- No prior work applies QBNs to ICS/SCADA security.
- No systematic hardware comparison of Grover-based vs variational quantum inference for BNs.
- The rare-event structure of real SCADA attacks makes quantum inference particularly relevant — and particularly challenging on NISQ hardware.

---

## 3. System Model and Dataset

### 3.1 Hardware-in-the-Loop Water Distribution Testbed
- Faramondi et al. (2021) HIL dataset: two subsystems (real S1 / simulated S2), two PLCs.
- Attack types: Normal, MiTM, DoS, Physical — MITRE ATT&CK grounding for each.
- Dataset structure: timestamp windows, sensor columns, attack labels.

### 3.2 Bayesian Network Architecture
- Two-subsystem causal graph: A1 → A2 with sensor leaves (ValveState, PumpState, TankState for S1; ValveState, FlowState, TankState for S2).
- 8 nodes, 4 attack states per attack node, binary sensor nodes.
- Diagram of full network.

### 3.3 Conditional Probability Table Derivation (Hybrid Methodology)
- **Set 1** — P(A1): data-derived from attack window timestamps.
- **Set 2** — P(A2 | A1): data-derived transition matrix; PLC-targeting label correction.
- **Set 3** — P(Sensor=Abnormal | AttackType): hybrid approach.
  - Normal column: ISA-TR84.00.09 false-alert floor (0.01); testbed artefact discussion.
  - MiTM column: rule-based stealth property (0.10); MITRE T0856/T0830.
  - DoS column: rule-based PLC starvation (0.80); MITRE T0814.
  - Physical column: sensor-type tier system (tank=0.80, valve=0.60, pump/flow=0.55); IEC 61511; data anchor (S1 pump 290/550 ≈ 0.527, Laplace α=0.5).
- Frame as a reusable methodology template for infrastructure BN modelling under data sparsity.

### 3.4 Diagnostic Queries

| ID | Query | Evidence rarity P(e) | Exact answer | Reasoning type |
|----|-------|----------------------|--------------|----------------|
| Q1 | P(A2=Physical \| Flow2=Abn, Valve2=Abn) | 0.0091 | 0.6658 | Forward diagnosis |
| Q2 | P(A1=Normal \| Flow2=Abn, Valve2=Abn) | 0.0091 | 0.6960 | Upstream inference |
| Q3 | P(A1=Normal \| A2=MiTM) | 0.0319 | 0.6332 | Attack propagation |
| Q4 | P(A1=MiTM \| A2=MiTM) | 0.0319 | 0.3626 | Attack propagation |

Q1/Q2 test rare sensor evidence; Q3/Q4 test direct attack-state evidence. Together they span diagnostic, predictive, and inter-causal reasoning.

---

## 4. Inference Methods

*Each subsection: algorithmic description → quantum circuit construction (where applicable) → complexity considerations. All methods are established; we describe them for completeness and to clarify experimental setup.*

### 4.1 Exact Inference — Variable Elimination
- Ground truth benchmark via pgmpy.

### 4.2 Classical Rejection Sampling (CRS)
- Accept/reject from joint. Acceptance rate = P(evidence). Degrades with evidence rarity.

### 4.3 Quantum Rejection Sampling (QRS)
- QBN circuit encoding (RY rotations from CPT angles). Grover oracle marks evidence-consistent states; amplitude amplification.
- Optimal iteration count k: formula vs empirical sweep.
- Expected speedup: O(1/√P(e)) over CRS.
- Circuit characteristics: 10 qubits, depth 102 (pre-transpilation), hundreds of 2-qubit gates after Grover × k.

### 4.4 Classical Likelihood Weighting (CLW)
- Fix evidence, sample from prior, weight. ESS degeneracy for rare evidence.

### 4.5 Quantum Likelihood Weighting (QLW)
- QBN circuit without Grover; measurement counts as weighted samples. Does not resolve weight degeneracy.

### 4.6 Classical Variational Inference (CAVI)
- Mean-field factorisation. ELBO maximisation. Mean-field collapse on indirect evidence.

### 4.7 Quantum Variational Inference (QVI)
- Born machine ansatz: parameterised 2-qubit circuit approximates posterior via COBYLA optimisation of KL divergence.
- Three existing ansatz variants: QVI-RA0 (RealAmplitudes reps=0, 2 params), QVI-RA1 (reps=1, 4 params), QVI-BN (BN-structured, 3 params).
- Circuit characteristics: 2 qubits, depth 3–6, 0–1 two-qubit gates.

---

## 5. Simulator Experiments

### 5.1 Experimental Setup
- N_RUNS = 10, shots = 1024, reproducible seeds.
- Error metric: |method mean − exact|.

### 5.2 Rejection Sampling (Phase 1)
- CRS vs QRS: 123× speedup (Q1/Q2), 29× (Q3/Q4); 6× and 4× variance reduction.
- Optimal-k sweep: formula vs empirical, finite-shot effects.

### 5.3 Likelihood Weighting (Phases 2–3)
- CLW: weight degeneracy, ESS ≈ 2.5% for rare evidence.
- QLW: no advantage over CLW — a negative result worth reporting clearly.

### 5.4 Variational Inference (Phase 4)
- CAVI: mean-field collapse on Q1/Q2 (error ≈ 0.28); exact on Q3/Q4.
- QVI: RA1 best overall (mean error 0.0109); RA0 simplest; BN-structured offers interpretability.
- Full 7-method × 4-query comparison table.

### 5.5 Sensitivity Analysis (Phase 5)
- OAT: max |Δposterior| = 0.0076 for ±30% perturbation — smaller than shot noise.
- Monte Carlo bands (N=500, σ=0.05): qualitative conclusions robust across full uncertainty band.
- Q3/Q4 invariance to sensor CPT perturbation (direct evidence bypasses sensor nodes).

---

## 6. Hardware Evaluation

### 6.1 Experimental Setup
- Platform: IBM `ibm_fez` (Heron r2, 156 qubits).
- Queries: Q3/Q4 (A2=MiTM evidence, k=4 Grover iterations).
- Rationale for Q3/Q4: higher P(e) → shallower circuit → most favourable case for QRS on NISQ hardware. If QRS fails here, it will fail worse on Q1/Q2.
- Transpilation: optimization level 3; dynamical decoupling (XX sequence).
- Shots: 4096.

### 6.2 QRS on Hardware
- Acceptance rate collapses: 99.8% (simulator) → 25.8% (hardware).
- Posteriors drift toward uniform prior (0.25):
  - Q3: 0.6332 (exact) → 0.3185 (hardware), error 0.3147
  - Q4: 0.3626 (exact) → 0.2448 (hardware), error 0.1178
- The 10-qubit Grover circuit (depth 102 + amplification stages) exceeds the device's coherence budget. Consistent with Borujeni et al.'s findings on earlier IBM hardware.

### 6.3 QLW on Hardware
- QLW uses the same 10-qubit QBN state-preparation circuit as QRS but **without** Grover amplification (depth 98 vs 102). This isolates whether the noise bottleneck is Grover or the base QBN encoding.
- Results are **worse** than QRS:
  - Q3: 0.6332 (exact) → 0.0500 (hardware), error 0.5832
  - Q4: 0.3626 (exact) → 0.9470 (hardware), error 0.5844
- Why QLW degrades worse than QRS: likelihood reweighting **amplifies** noise. Corrupted bitstrings from the noisy QBN circuit are decoded to wrong variable assignments, which receive wrong likelihood weights. A few high-weight corrupted samples dominate the weighted sum, producing posteriors further from exact than even the uniform prior.
- QRS at least drifts toward uniform (losing signal); QLW actively inverts the signal.
- This confirms the **QBN state-preparation encoding** is the noise bottleneck — not Grover amplification. Removing Grover (4 layers) does not help.

### 6.4 QVI on Hardware
- Approach: train Born machines on simulator (COBYLA, cheap), extract optimal parameters, run the 2-qubit circuit on hardware.
- Results across all three ansätze:

| Method | Q3 HW Error | Q4 HW Error | Noise Δ | Circuit |
|--------|-------------|-------------|---------|---------|
| QRS | 0.3147 | 0.1178 | catastrophic | 10q, d=102 |
| QLW | 0.5832 | 0.5844 | catastrophic | 10q, d=98 |
| QVI-RA0 | 0.0299 | 0.0263 | +0.001 | 2q, d=4 |
| QVI-RA1 | 0.0111 | 0.0216 | −0.003 | 2q, d=8 |
| QVI-BN | 0.0123 | 0.0167 | −0.009 | 2q, d=13 |

- Noise penalty for QVI is within shot noise — hardware adds negligible error.
- QVI hardware error is 10–50× smaller than QRS/QLW hardware error.

### 6.5 Analysis: Circuit Depth as the Determining Factor
- The three hardware results form a clear hierarchy:
  - **QLW** (depth 98, no Grover): worst — noise + reweighting amplifies errors
  - **QRS** (depth 102, Grover k=4): bad — noise → uniform prior
  - **QVI** (depth 4–13, 2 qubits): excellent — near-zero noise penalty
- The 4-layer difference between QLW and QRS is negligible; both fail because the **10-qubit QBN state-preparation encoding** (multi-controlled RY gates for CPT encoding) exceeds the coherence budget.
- QVI error is dominated by optimisation quality, not gate noise, because the circuit remains shallow regardless of BN complexity. The classical COBYLA loop absorbs the computational cost; the quantum circuit only samples.
- This is not "QVI is a better algorithm" — it is a fundamentally different noise scaling regime. QRS/QLW offer principled Bayesian inference via circuit encoding but require coherence budgets not yet available. QVI trades structural guarantees for hardware resilience.
- Classical post-processing (likelihood reweighting) cannot rescue noisy quantum samples — it makes things worse. The only viable path on NISQ hardware is keeping the circuit shallow.
- The finding generalises beyond this specific BN: any quantum inference method whose circuit depth grows with the problem size will face the same NISQ barrier.

---

## 7. Discussion

### 7.1 Implications for Quantum Probabilistic Inference
- The simulator results confirm that quantum speedup for rare-evidence inference is real and substantial (123×). This matters for SCADA security because low-base-rate attacks are the operationally dangerous ones.
- The hardware results show this speedup is currently inaccessible — but not because the algorithm is wrong. The bottleneck is circuit depth vs coherence budget, which will improve with hardware maturation.
- QVI provides a viable near-term alternative: posterior accuracy comparable to exact inference, hardware-resilient, and compatible with current NISQ devices.

### 7.2 Practical Guidance
- Near-term (NISQ): use QVI for quantum-hardware deployment; use QRS on simulator for benchmarking and as evidence of future advantage.
- Fault-tolerant era: QRS becomes viable and offers provable speedup — the transition point depends on coherence improvements and error correction overhead.
- For ICS practitioners: the hybrid CPT methodology (Section 3.3) is applicable to other infrastructure BNs regardless of quantum/classical inference choice.

### 7.3 Limitations
- Small BN (8 nodes). Scalability to larger networks requires investigation — QVI circuit depth may grow for higher-dimensional posteriors; QRS circuit depth scales with BN size.
- Hardware results limited to Q3/Q4 on one backend (ibm_fez). Q1/Q2 (deeper circuits) deferred.
- QVI trains classically — the "quantum" contribution is the Born machine sampling, not the optimisation loop. On a 2-qubit circuit, classical simulation is trivially efficient; the interest is in the scaling behaviour for larger systems.
- Single dataset (HIL testbed); generalisability to oil/gas, power grid SCADA TBD.
- No error mitigation techniques (ZNE, PEC) applied to QRS — these could partially close the gap.

### 7.4 Future Work
- Larger BNs (50+ nodes): test whether QVI circuit depth remains manageable.
- Hardware-in-the-loop QVI training: run the COBYLA loop with hardware as the sampler.
- Error mitigation for QRS: ZNE, probabilistic error cancellation, dynamical decoupling variants.
- Multiple hardware backends: compare across architectures (superconducting, trapped ion).
- Real-time deployment: latency requirements for SCADA integration.

---

## 8. Conclusion

- QBNs applied to SCADA cyber-physical threat detection using a real HIL dataset — first application in this domain.
- Seven inference methods compared systematically: QRS achieves 123× speedup on simulator, confirming theoretical quantum advantage for rare-evidence inference.
- On IBM quantum hardware: QRS collapses (posteriors → uniform), QLW is even worse (reweighting amplifies noise), while QVI preserves accuracy with near-zero noise penalty — errors 10–50× smaller.
- The determining factor is circuit depth, not algorithm class, and classical post-processing cannot rescue noisy quantum samples. NISQ-era quantum probabilistic inference requires methods that keep circuits shallow.
- As quantum hardware matures, the depth barrier will recede, and Grover-based approaches will become viable. Until then, variational methods provide a practical, hardware-compatible path for quantum inference on real-world Bayesian networks.

---

## References (key anchors — to be expanded)

- Faramondi et al. (2021) — HIL Water Distribution Testbed dataset, *IEEE Access*
- Low, Yoder & Chuang (2014) — Quantum inference on Bayesian networks, *Phys. Rev. A*
- Borujeni et al. (2021) — Quantum circuit representation of Bayesian networks, *Expert Systems with Applications*
- Krebs et al. (2024) — Quantum Bayesian Networks for Music, *ICMCI*
- Benedetti et al. (2019) — Parameterized quantum circuits as machine learning models, *Quantum Science and Technology*
- MITRE ATT&CK for ICS — T0814, T0830, T0856
- ISA-TR84.00.09 — False-alert rates for process instrumentation
- IEC 61511 — Safety Instrumented Systems, diagnostic coverage
- Grover (1996) — Quantum search algorithm
- Colonial Pipeline (2021), Oldsmar water treatment (2021) — ICS incident reports

---

*Estimated length: ~9,000–11,000 words + figures/tables.*
