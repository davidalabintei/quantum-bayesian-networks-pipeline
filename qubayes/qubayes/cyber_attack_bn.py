import numpy as np
from qubayes.qubayes_tools import Graph, Node, BayesNet

# ── CPT constants — single source of truth ────────────────────────────────
# Update these when probabilities are revised; both the QRS notebook and the
# variational inference code import from here.

ATTACK_ORDER = ["Normal", "MiTM", "DoS", "Physical"]

# Set 1: P(AttackType_1)  [Normal, MiTM, DoS, Physical]
P_A1 = np.array([0.87711808, 0.01214143, 0.03735824, 0.07338225])

# Set 2: P(AttackType_2 | AttackType_1)  rows=A2, cols=A1
# S2 = PLC2-controlled interface (T6, V20, Flow1). PLC3/PLC4 excluded from model.
# Fix: cyber attack labelling uses primary PLC target only; DoS cascade from
# PLC1 to S2 via effect text is removed (S2 PLCs are not under DoS when PLC1 is flooded).
A21 = np.array([
    [0.9561, 0.0376, 0.8918, 0.9973],   # A2=Normal
    [0.0230, 0.9516, 0.0018, 0.0009],   # A2=MiTM
    [0.0001, 0.0054, 0.1046, 0.0009],   # A2=DoS
    [0.0208, 0.0054, 0.0018, 0.0009],   # A2=Physical
])

# Set 3: P(Sensor=Abnormal | AttackType)  [Normal, MiTM, DoS, Physical]
#
# Hybrid methodology — four columns derived by different means:
#
# ── Normal column (0.01 for all sensors) ───────────────────────────────────────
#   The HIL dataset shows zero false positives across all Normal-window timestamps
#   (0/6574 in S1, 0/7091 in S2). However, a literal zero is a testbed artefact:
#   real deployments have calibration drift, measurement noise, and communication
#   glitches. A floor of 0.01 is used, consistent with ISA-TR84.00.09 false-alert
#   rates for well-calibrated process instrumentation in water infrastructure.
#
# ── MiTM column (0.10 for all sensors) ─────────────────────────────────────────
#   Man-in-the-middle attacks (ARP spoofing, false data injection) are designed to
#   be stealthy: the attacker replays or injects legitimate-looking values so the
#   SCADA operator sees no anomaly while the physical process deviates. In this BN
#   the sensor nodes represent SCADA-reported readings, so P(Abn|MiTM) is low.
#   Value = 0.10: not zero because no spoofing is perfect — timing jitter, polling
#   mismatches, and partial packet loss can produce brief detectable inconsistencies.
#   Grounded in MITRE ATT&CK for ICS: T0856 (Spoof Reporting Message),
#   T0830 (Manipulate I/O Image); and Liu et al. (2011, IEEE Trans. Inf. Theory) on
#   stealthy false data injection. Applied uniformly across all sensor types because
#   the stealth rule operates at the PLC/network level, not the sensor level.
#
# ── DoS column (0.80 for all sensors) ──────────────────────────────────────────
#   Denial-of-service attacks (ICMP/TCP floods) starve the PLC so it stops issuing
#   control commands. The uncontrolled physical process drifts from setpoint, and
#   sensor readings eventually become abnormal (stale values, fail-safe defaults, or
#   out-of-range process variables). P = 0.80, not 1.0, because PLCs have watchdog
#   timers that may hold the last output for a grace period, and physical processes
#   have inertia — brief DoS bursts may not produce a detectable deviation within
#   every timestamp. Grounded in MITRE ATT&CK for ICS: T0814 (Denial of Service);
#   Ralston et al. (2007, ISA Transactions) on observable process deviation under
#   PLC starvation. Applied uniformly across sensor types for the same reason as
#   MiTM — the rule is at the PLC level.
#
# ── Physical column — data-informed, sensor-type tier system ───────────────────
#   Physical faults are component-localised, so emission probabilities are assigned
#   by the functional role of each sensor type relative to typical physical faults
#   in a water distribution subsystem. Three tiers are defined:
#
#   Primary   (sensor directly monitors the faulty component):   P = 0.85
#     — not used here; reserved for single-component BN extensions
#   Tank      (integrating state variable — accumulates any subsystem fault): P = 0.80
#     Justification: tank level integrates all mass flows; any fault (pump failure,
#     valve fault, pipe leak) will eventually manifest in tank level. Not 1.0 because
#     some faults are detected and isolated before the level exceeds the anomaly
#     threshold. Consistent with IEC 61511 diagnostic coverage for level sensors
#     (0.90–0.99 for safety-rated instruments; lower for research testbeds).
#     Data anchor: S1_tank raw count = 550/550 (Laplace-smoothed) reflects the
#     labelling artefact of perfect window coverage; 0.80 corrects for process
#     inertia and detection lag.
#   Valve/Flow (actuator and downstream process variable):       P = 0.60 / 0.55
#     Valves can be both fault targets and control responses; flow sensors are
#     downstream of valve/pump changes. Moderate probability reflects the bidirectional
#     role. IEC 61511 diagnostic coverage for valve position: 0.60–0.90.
#     Consistent across S1 and S2 because both subsystems contain hydraulic valves
#     and flow paths of the same physical type. S1_valve is NOT set to near-zero
#     (data artefact) because valve faults are equally possible in S1 — the HIL
#     dataset simply did not include that scenario.
#   Pump      (mechanical component, fault-specific):            P = 0.55
#     Data anchor: S1_pump raw count = 290/550 ≈ 0.527 (data-derived, Laplace α=0.5).
#     The 550 denominator includes all S1 physical fault types, diluting the pump
#     signal; 0.55 preserves the empirical anchor while rounding to a tier value.

# Subsystem 1
P_VALVE_S1 = np.array([0.01, 0.10, 0.80, 0.60])  # valve: Normal=ISA floor, MiTM=stealth, DoS=PLC starvation, Phys=valve/actuator tier
P_PUMP_S1  = np.array([0.01, 0.10, 0.80, 0.55])  # pump:  data anchor 290/550≈0.527 → 0.55 tier
P_TANK_S1  = np.array([0.01, 0.10, 0.80, 0.80])  # tank:  integrating variable tier

# Subsystem 2  (no pump in S2)
P_VALVE_S2 = np.array([0.01, 0.10, 0.80, 0.60])  # valve: same type as S1_valve — consistent tier
P_FLOW_S2  = np.array([0.01, 0.10, 0.80, 0.55])  # flow:  downstream process variable, same tier as pump
P_TANK_S2  = np.array([0.01, 0.10, 0.80, 0.80])  # tank:  integrating variable tier — same as S1_tank


class CyberAttackA2BN(BayesNet):
    """
    Variational inference wrapper for the cyber QBN, focusing on the
    posterior over AttackType_2 given evidence on subsystem-2 sensors.

    Latent variable z: AttackType_2 (4 states) encoded as 2 bits:
        b1 b0 | state
         0  0 | Normal
         0  1 | MiTM
         1  0 | DoS
         1  1 | Physical
    """

    def __init__(self):
        self.attack_order   = ATTACK_ORDER
        self.attack_to_idx  = {k: i for i, k in enumerate(ATTACK_ORDER)}

        self.P_A1       = P_A1
        self.A21        = A21
        self.p_valve_s2 = P_VALVE_S2
        self.p_flow_s2  = P_FLOW_S2
        self.p_tank_s2  = P_TANK_S2

        assert np.allclose(self.A21.sum(axis=0), 1.0), "A21 columns must sum to 1."
        self.graph = self._create_graph()

    def _create_graph(self):
        """Minimal placeholder graph for framework compatibility."""
        a2 = Node("AttackType_2", data=np.ones(4) / 4.0)
        return Graph({"AttackType_2": a2})

    # ── 2-bit encoding helpers ────────────────────────────────────────────

    @staticmethod
    def a2_idx_to_bits(a2_idx: int):
        """Map attack state index (0-3) to (b0, b1) bit pair."""
        return (a2_idx & 1, (a2_idx >> 1) & 1)

    @staticmethod
    def bits_to_a2_idx(bits_2):
        """Map (b0, b1) bit pair back to attack state index."""
        b0, b1 = int(bits_2[0]), int(bits_2[1])
        return b0 + 2 * b1

    def _marginal_P_A2(self):
        """P(A2=j) = sum_i P(A2=j | A1=i) * P(A1=i)"""
        return (self.A21 * self.P_A1.reshape(1, 4)).sum(axis=1)

    # ── BayesNet API ──────────────────────────────────────────────────────

    def compute_p_prior(self):
        """
        Return p(z) as a (2, 2) tensor over the 2-bit latent variable.
        """
        p_a2    = self._marginal_P_A2()
        p_bits  = np.zeros((2, 2), dtype=float)
        for a2_idx in range(4):
            b0, b1 = self.a2_idx_to_bits(a2_idx)
            p_bits[b0, b1] = p_a2[a2_idx]
        return p_bits / p_bits.sum()

    def sample_from_prior(self, n_samples: int):
        """
        Draw n_samples from p(z).  Returns array of shape (n_samples, 2).
        """
        p_a2    = self._marginal_P_A2()
        a2_draws = np.random.choice(4, size=n_samples, p=p_a2 / p_a2.sum())
        bits     = np.array([self.a2_idx_to_bits(i) for i in a2_draws], dtype=np.int32)
        return bits

    def compute_log_likelihood(self, samples, evidence=None):
        """
        Log p(evidence | z) for each sample.

        Parameters
        ----------
        samples  : (n_samples, 2) int array — bit pairs for AttackType_2
        evidence : dict mapping sensor name to int (1=Abnormal, 0=Normal)
                   Supported keys: FlowState_2, ValveState_2, TankState_2
        """
        if not evidence:
            return np.zeros(samples.shape[0])

        sensor_map = {
            "ValveState_2": self.p_valve_s2,
            "FlowState_2":  self.p_flow_s2,
            "TankState_2":  self.p_tank_s2,
        }

        log_lik = np.zeros(samples.shape[0])
        for i, bits in enumerate(samples):
            a2_idx = self.bits_to_a2_idx(bits)
            ll = 0.0
            for node, val in evidence.items():
                if node not in sensor_map:
                    raise ValueError(f"Unsupported evidence node: {node}")
                p_ab = sensor_map[node][a2_idx]
                p    = p_ab if int(val) == 1 else (1.0 - p_ab)
                ll  += np.log(max(1e-12, p))
            log_lik[i] = ll
        return log_lik

    def compute_posterior(self, evidence=None):
        """
        Exact posterior p(z | evidence) as a (2, 2) tensor.
        Used for TVD reporting in the variational inference script.
        """
        p_bits = self.compute_p_prior()
        p_a2   = np.array([p_bits[self.a2_idx_to_bits(i)] for i in range(4)])

        if evidence:
            sensor_map = {
                "ValveState_2": self.p_valve_s2,
                "FlowState_2":  self.p_flow_s2,
                "TankState_2":  self.p_tank_s2,
            }
            lik = np.ones(4)
            for a2_idx in range(4):
                for node, val in evidence.items():
                    if node not in sensor_map:
                        raise ValueError(f"Unsupported evidence node: {node}")
                    p_ab   = sensor_map[node][a2_idx]
                    p      = p_ab if int(val) == 1 else (1.0 - p_ab)
                    lik[a2_idx] *= max(1e-12, p)
            p_a2 = p_a2 * lik

        p_a2 /= p_a2.sum()

        post_bits = np.zeros((2, 2))
        for a2_idx in range(4):
            b0, b1 = self.a2_idx_to_bits(a2_idx)
            post_bits[b0, b1] = p_a2[a2_idx]
        return post_bits


class CyberAttackA1BN(BayesNet):
    """
    Variational inference wrapper targeting AttackType_1 posterior.

    Latent variable z: AttackType_1 (4 states) encoded as 2 bits:
        b1 b0 | state
         0  0 | Normal   (0)
         0  1 | MiTM     (1)
         1  0 | DoS      (2)
         1  1 | Physical (3)

    Supports two evidence modes (auto-detected from the evidence dict):
        Sensor evidence  — keys are sensor node names (e.g. FlowState_2)
                           P(e | A1=i) = Σ_{a2} P(e|A2=a2) · A21[a2, i]
        Direct A2        — key is 'AttackType_2'
                           P(A2=obs | A1=i) = A21[obs, i]
    """

    _ALL_SENSORS = {
        "ValveState_2": P_VALVE_S2, "FlowState_2": P_FLOW_S2,
        "TankState_2":  P_TANK_S2,  "ValveState_1": P_VALVE_S1,
        "PumpState_1":  P_PUMP_S1,  "TankState_1":  P_TANK_S1,
    }

    def __init__(self):
        self.attack_order  = ATTACK_ORDER
        self.attack_to_idx = {k: i for i, k in enumerate(ATTACK_ORDER)}
        self.P_A1 = P_A1
        self.A21  = A21
        self.graph = self._create_graph()

    def _create_graph(self):
        a1 = Node("AttackType_1", data=np.ones(4) / 4.0)
        return Graph({"AttackType_1": a1})

    @staticmethod
    def a1_idx_to_bits(a1_idx: int):
        return (a1_idx & 1, (a1_idx >> 1) & 1)

    @staticmethod
    def bits_to_a1_idx(bits_2):
        return int(bits_2[0]) + 2 * int(bits_2[1])

    def _log_lik_per_state(self, evidence: dict) -> np.ndarray:
        """Shape-(4,) array: log P(evidence | A1=i) for i = 0..3."""
        if "AttackType_2" in evidence:
            a2_obs = int(evidence["AttackType_2"])
            return np.log(np.maximum(self.A21[a2_obs, :], 1e-12))

        # Sensor evidence: marginalise over A2
        p_e_given_a2 = np.ones(4, dtype=float)
        for node, val in evidence.items():
            if node not in self._ALL_SENSORS:
                raise ValueError(f"Unsupported evidence node: {node}")
            p_ab = self._ALL_SENSORS[node]
            for j in range(4):
                p = float(p_ab[j]) if int(val) == 1 else (1.0 - float(p_ab[j]))
                p_e_given_a2[j] *= max(1e-12, p)
        # P(e | A1=i) = Σ_j P(e | A2=j) · A21[j, i]
        return np.log(np.maximum(p_e_given_a2 @ self.A21, 1e-12))

    def compute_p_prior(self) -> np.ndarray:
        """P(A1) as a (2, 2) tensor over (b0, b1) bit pairs."""
        p_bits = np.zeros((2, 2), dtype=float)
        for i in range(4):
            b0, b1 = self.a1_idx_to_bits(i)
            p_bits[b0, b1] = self.P_A1[i]
        return p_bits / p_bits.sum()

    def sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Draw n_samples from P(A1). Returns (n_samples, 2) int array."""
        a1_draws = np.random.choice(4, size=n_samples,
                                    p=self.P_A1 / self.P_A1.sum())
        return np.array([self.a1_idx_to_bits(i) for i in a1_draws],
                        dtype=np.int32)

    def compute_log_likelihood(self, samples, evidence=None) -> np.ndarray:
        """log P(evidence | A1) for each (b0, b1) sample row."""
        if not evidence:
            return np.zeros(samples.shape[0])
        table = self._log_lik_per_state(evidence)
        return np.array([table[self.bits_to_a1_idx(s)] for s in samples])

    def compute_posterior(self, evidence=None) -> np.ndarray:
        """Exact P(A1 | evidence) as (2, 2) tensor. Used for TVD tracking."""
        log_p = np.log(np.maximum(self.P_A1, 1e-12))
        if evidence:
            log_p = log_p + self._log_lik_per_state(evidence)
        log_p -= log_p.max()
        p_a1 = np.exp(log_p); p_a1 /= p_a1.sum()
        post_bits = np.zeros((2, 2), dtype=float)
        for i in range(4):
            b0, b1 = self.a1_idx_to_bits(i)
            post_bits[b0, b1] = p_a1[i]
        return post_bits
