import numpy as np
from qubayes.qubayes_tools import Graph, Node, BayesNet

# ── CPT constants — single source of truth ────────────────────────────────
# Update these when probabilities are revised; both the QRS notebook and the
# variational inference code import from here.

ATTACK_ORDER = ["Normal", "MiTM", "DoS", "Physical"]

# Set 1: P(AttackType_1)  [Normal, MiTM, DoS, Physical]
P_A1 = np.array([0.87711808, 0.01214143, 0.03735824, 0.07338225])

# Set 2: P(AttackType_2 | AttackType_1)  rows=A2, cols=A1
A21 = np.array([
    [0.9561, 0.0376, 0.1259, 0.9973],   # A2=Normal
    [0.0230, 0.9516, 0.0018, 0.0009],   # A2=MiTM
    [0.0001, 0.0054, 0.8705, 0.0009],   # A2=DoS
    [0.0208, 0.0054, 0.0018, 0.0009],   # A2=Physical
])

# Set 3: P(Sensor=Abnormal | AttackType)  [Normal, MiTM, DoS, Physical]
# Subsystem 1
P_VALVE_S1 = np.array([0.00, 0.01, 0.05, 0.99])
P_PUMP_S1  = np.array([0.00, 0.05, 0.10, 0.55])
P_TANK_S1  = np.array([0.00, 0.01, 0.05, 0.99])

# Subsystem 2  (no pump in S2)
P_VALVE_S2 = np.array([0.00, 0.01, 0.05, 0.99])
P_FLOW_S2  = np.array([0.00, 0.05, 0.05, 0.45])
P_TANK_S2  = np.array([0.00, 0.01, 0.05, 0.99])


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
