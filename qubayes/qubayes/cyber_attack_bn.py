import numpy as np
from qubayes.qubayes.qubayes_tools import Graph, Node, BayesNet


class CyberAttackA2BN(BayesNet):
    """
    Variational inference wrapper for the cyber QBN, focusing on
    posterior over AttackType_2 given evidence on subsystem-2 emissions.

    Latent variable z: AttackType_2 (4 states) encoded as 2 bits:
        00 Normal
        01 MiTM
        10 DoS
        11 Physical
    """

    def __init__(self):
        # --- Attack state order must match your QBN code ---
        self.attack_order = ["Normal", "MiTM", "DoS", "Physical"]
        self.attack_to_idx = {k: i for i, k in enumerate(self.attack_order)}

        # --- Locked priors from your rejection sampling code ---
        self.P_A1 = np.array([0.87711808, 0.01214143, 0.03735824, 0.07338225], dtype=float)

        self.A21 = np.array([
            [0.9561, 0.0376, 0.1259, 0.9973],  # A2=Normal
            [0.0230, 0.9516, 0.0018, 0.0009],  # A2=MiTM
            [0.0001, 0.0054, 0.8705, 0.0009],  # A2=DoS
            [0.0208, 0.0054, 0.0018, 0.0009],  # A2=Physical
        ], dtype=float)

        if not np.allclose(self.A21.sum(axis=0), 1.0):
            raise ValueError(f"A21 columns must sum to 1. Got {self.A21.sum(axis=0)}")

        # --- Locked emission probabilities for subsystem 2 ---
        # These are P(sensor=Abnormal | AttackType_2 = state)
        # Order must match attack_order above.
        self.p_valve_s2 = np.array([0.00, 0.01, 0.05, 0.99], dtype=float)
        self.p_flow_s2  = np.array([0.00, 0.05, 0.05, 0.45], dtype=float)
        self.p_tank_s2  = np.array([0.00, 0.01, 0.05, 0.99], dtype=float)

        # Keep a graph object for compatibility, though VI below only uses
        # A2 marginal and emissions.
        self.graph = self.create_graph()

    def create_graph(self):
        # Node states here are numeric for the VI flow.
        # AttackType_2 is represented as 2 bits externally in VI; graph is minimal.
        a2 = Node("AttackType_2", data=np.ones(4) / 4.0)  # placeholder
        bn = Graph({"AttackType_2": a2})
        return bn

    # -----------------------------
    # Helpers for 2-bit encoding
    # -----------------------------
    @staticmethod
    def a2_idx_to_bits(a2_idx: int):
        # 0..3 -> (b0,b1) with b0 = LSB if you want, but be consistent everywhere.
        # We'll use (b0,b1) = binary of idx with b0 = least significant bit.
        return (a2_idx & 1, (a2_idx >> 1) & 1)

    @staticmethod
    def bits_to_a2_idx(bits_2):
        b0, b1 = int(bits_2[0]), int(bits_2[1])
        return b0 + 2 * b1

    def _marginal_P_A2(self):
        # P(A2=j) = sum_i P(A2=j | A1=i) P(A1=i)
        # A21 is [A2, A1]
        return (self.A21 * self.P_A1.reshape(1, 4)).sum(axis=1)

    # -----------------------------
    # BayesNet API used by VI code
    # -----------------------------
    def compute_p_prior(self):
        """
        Return p(z) as a tensor over the latent bit variables.
        Here z = 2 bits representing AttackType_2, so shape is (2,2).
        """
        p_a2 = self._marginal_P_A2()
        p_bits = np.zeros((2, 2), dtype=float)
        for a2_idx in range(4):
            b0, b1 = self.a2_idx_to_bits(a2_idx)
            p_bits[b0, b1] = p_a2[a2_idx]
        return p_bits / p_bits.sum()

    def sample_from_prior(self, n_samples: int):
        """
        Draw samples of z (2 bits) from p(z).
        Returns array shape (n_samples, 2) with entries in {0,1}.
        """
        p_a2 = self._marginal_P_A2()
        a2_draws = np.random.choice(4, size=n_samples, p=p_a2 / p_a2.sum())
        out = np.zeros((n_samples, 2), dtype=np.int32)
        for i, a2_idx in enumerate(a2_draws):
            out[i, :] = np.array(self.a2_idx_to_bits(a2_idx), dtype=np.int32)
        return out

    def compute_log_likelihood(self, samples, evidence=None):
        """
        samples: (n_samples, 2) bits for AttackType_2
        evidence: dict like {"FlowState_2": 1, "ValveState_2": 1, "TankState_2": 0}
                  where 1 means Abnormal, 0 means Normal.

        Returns log p(evidence | z) per sample.
        """
        if evidence is None or len(evidence) == 0:
            return np.zeros((samples.shape[0],), dtype=float)

        log_lik = np.zeros((samples.shape[0],), dtype=float)

        for i in range(samples.shape[0]):
            a2_idx = self.bits_to_a2_idx(samples[i, :])

            ll = 0.0
            for node, val in evidence.items():
                val = int(val)

                if node == "ValveState_2":
                    p_ab = self.p_valve_s2[a2_idx]
                elif node == "FlowState_2":
                    p_ab = self.p_flow_s2[a2_idx]
                elif node == "TankState_2":
                    p_ab = self.p_tank_s2[a2_idx]
                else:
                    raise ValueError(f"Unsupported evidence node for this VI wrapper: {node}")

                p = p_ab if val == 1 else (1.0 - p_ab)

                # Avoid log(0) without injecting eps into the model probabilities
                # This is only numerical protection for the log operation.
                ll += np.log(max(1e-12, p))

            log_lik[i] = ll

        return log_lik

    def compute_posterior(self, evidence=None):
        """
        Exact posterior over z bits, shape (2,2).
        Used only for reporting TVD in your VI script.
        """
        p_prior_bits = self.compute_p_prior()

        # Convert bit-tensor to a2 prior vector, apply likelihood, convert back
        p_a2 = np.zeros(4, dtype=float)
        for a2_idx in range(4):
            b0, b1 = self.a2_idx_to_bits(a2_idx)
            p_a2[a2_idx] = p_prior_bits[b0, b1]

        if evidence is None or len(evidence) == 0:
            post_a2 = p_a2 / p_a2.sum()
        else:
            lik = np.ones(4, dtype=float)
            for a2_idx in range(4):
                L = 1.0
                for node, val in evidence.items():
                    val = int(val)
                    if node == "ValveState_2":
                        p_ab = self.p_valve_s2[a2_idx]
                    elif node == "FlowState_2":
                        p_ab = self.p_flow_s2[a2_idx]
                    elif node == "TankState_2":
                        p_ab = self.p_tank_s2[a2_idx]
                    else:
                        raise ValueError(f"Unsupported evidence node for this VI wrapper: {node}")
                    p = p_ab if val == 1 else (1.0 - p_ab)
                    L *= max(1e-12, p)
                lik[a2_idx] = L
            post_a2 = (p_a2 * lik)
            post_a2 = post_a2 / post_a2.sum()

        post_bits = np.zeros((2, 2), dtype=float)
        for a2_idx in range(4):
            b0, b1 = self.a2_idx_to_bits(a2_idx)
            post_bits[b0, b1] = post_a2[a2_idx]
        return post_bits
