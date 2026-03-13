# from qubayes.qubayes.qubayes_tools import Node, Graph, BayesNet
# import numpy as np

# class RailAllBN(BayesNet):
#     # def __init__(self):
#         # self.target = {"RailAll": "True"}

#     @staticmethod
#     def create_graph():
#         # Root node
#         railall = Node("RailAll", data=np.array([0.18, 0.82]), states={"True": 0, "False": 1})

#         # Each child has shape (2,): P(child | RailAll=True/False)
#         def bin_cpt(p_true_given_true, p_true_given_false):
#             return np.array([
#                 [p_true_given_true, p_true_given_false],  # P(True | RailAll)
#                 [1 - p_true_given_true, 1 - p_true_given_false]  # P(False | RailAll)
#             ])

#         alignment = Node("Alignment", parents=["RailAll"],
#                          data=bin_cpt(0.054, 0.001), states={"True": 0, "False": 1})
#         crosslevel = Node("Crosslevel", parents=["RailAll"],
#                           data=bin_cpt(0.30, 0.003), states={"True": 0, "False": 1})
#         warp62 = Node("Warp62", parents=["RailAll"],
#                       data=bin_cpt(0.095, 0.001), states={"True": 0, "False": 1})
#         warp31 = Node("Warp31", parents=["RailAll"],
#                       data=bin_cpt(0.22, 0.002), states={"True": 0, "False": 1})
#         gage = Node("Gage", parents=["RailAll"],
#                     data=bin_cpt(0.15, 0.002), states={"True": 0, "False": 1})
#         cant = Node("Cant", parents=["RailAll"],
#                     data=bin_cpt(0.37, 0.004), states={"True": 0, "False": 1})

#         graph = Graph({
#             "RailAll": railall,
#             "Alignment": alignment,
#             "Crosslevel": crosslevel,
#             "Warp62": warp62,
#             "Warp31": warp31,
#             "Gage": gage,
#             "Cant": cant
#         })
#         return graph

#     def __init__(self):
#         super().__init__(self.create_graph())
#         self.target = {"RailAll": "True"}
#         self.target_node = list(self.target.keys())[0]
#         self.target_state = self.target[self.target_node]
        
#         # Get index of the target state
#         target_idx = self.graph.nodes[self.target_node].states[self.target_state]
        
#         # Compute how many bits are needed to encode the target node
#         n_states = len(self.graph.nodes[self.target_node].states)
#         n_bits = int(np.ceil(np.log2(n_states)))
        
#         # Manually create the bitstring for the target state
#         self.bitstring = format(target_idx, f"0{n_bits}b")


#     # def compute_p_prior(self):
#     #     return self.graph.marginalize_all_but(["RailAll"])
#     # def compute_p_prior(self):
#     #     return self.graph.nodes["RailAll"].data
#     def compute_p_prior(self):
#         self.p_prior = self.graph.marginalize_all_but(["RailAll"])
#         return self.p_prior

#     def compute_tvd(self, samples):
#         # Count sample occurrences
#         counts = {}
#         for s in samples:
#             idx = int(s[0])  # Only RailAll bitstring
#             counts[idx] = counts.get(idx, 0) + 1
    
#         total = len(samples)
#         empirical = np.zeros_like(self.p_prior)
#         for idx, count in counts.items():
#             empirical[idx] = count / total
    
#         # Ensure flattening (if needed)
#         p_true = self.p_prior.flatten() if isinstance(self.p_prior, np.ndarray) else self.p_prior
    
#         return 0.5 * np.sum(np.abs(empirical - p_true))


#     def compute_log_likelihood(self, samples, observed_node="Crosslevel", observed_value=0):
#         # samples are shape (N, 1) for RailAll (1 qubit)
#         # output: log P(observed_node = value | RailAll)
#         log_lik = np.zeros((samples.shape[0],))
#         for i in range(samples.shape[0]):
#             railall_state = samples[i, 0]
#             prob = self.graph.nodes[observed_node].data[observed_value, railall_state]
#             log_lik[i] = np.log(max(prob, 1e-5))
#         return log_lik

    
#     # def sample_from_prior(self, n_samples=100):
#     #     probs = self.graph.nodes["RailAll"].data
#     #     samples = np.random.choice([0, 1], size=n_samples, p=probs)
#     #     return samples[:, np.newaxis]
#     def sample_from_prior(self, n_samples=100):
#         # Sample from the prior P(RailAll)
#         samples = np.random.choice([0, 1], size=(n_samples, 1), p=self.graph.nodes["RailAll"].data)
#         return samples
#     # def compute_tvd(self, samples):
#     # # Count frequencies from samples
#     #     counts = {}
#     #     for s in samples:
#     #         idx = int(s[0])  # assuming RailAll is 1D
#     #         counts[idx] = counts.get(idx, 0) + 1
    
#     #     total = len(samples)
#     #     empirical = np.zeros_like(self.p_prior)
#     #     for idx, count in counts.items():
#     #         empirical[idx] = count / total
    
#     #     # True prior from earlier method
#     #     p_true = self.p_prior
    
#     #     # Ensure both are 1D numpy arrays
#     #     if isinstance(p_true, np.ndarray) and len(p_true.shape) > 1:
#     #         p_true = p_true.flatten()
    
#     #     return 0.5 * np.sum(np.abs(empirical - p_true))
##################################
# import numpy as np
# from qubayes.qubayes.qubayes_tools import Query, QBNQuery, Node, Graph, BayesNet


# class RailAllBN(BayesNet):

#     def __init__(self, random_cpd=True):
#         self.random_cpd = random_cpd
#         self.graph = self.create_graph()
#         if random_cpd:
#             self.graph.nodes['railall'].data[0] = np.random.uniform(low=0.01, high=0.99)
#             self.graph.nodes['railall'].data[1] = 1. - self.graph.nodes['railall'].data[0]
#             for node in ['alignment', 'crosslevel', 'warp62', 'warp31', 'gage', 'cant']:
#                 self.graph.nodes[node].data[0, 0] = np.random.uniform(low=0.01, high=0.99)
#                 self.graph.nodes[node].data[1, 0] = 1. - self.graph.nodes[node].data[0, 0]
#                 self.graph.nodes[node].data[0, 1] = np.random.uniform(low=0.01, high=0.99)
#                 self.graph.nodes[node].data[1, 1] = 1. - self.graph.nodes[node].data[0, 1]

#     @staticmethod
#     def create_graph():
#         railall = Node('railall', data=np.array([0.18, 0.82]))  # True: 0.18, False: 0.82
#         alignment = Node('alignment', data=np.array([[0.946, 0.054],  # P(False|railall=False), P(False|railall=True)
#                                                     [0.054, 0.946]]),  # P(True|railall=False), P(True|railall=True)
#                          parents=['railall'])
#         crosslevel = Node('crosslevel', data=np.array([[0.70, 0.30],  # P(False|railall=False), P(False|railall=True)
#                                                       [0.30, 0.70]]),  # P(True|railall=False), P(True|railall=True)
#                           parents=['railall'])
#         warp62 = Node('warp62', data=np.array([[0.905, 0.095],  # P(False|railall=False), P(False|railall=True)
#                                               [0.095, 0.905]]),  # P(True|railall=False), P(True|railall=True)
#                       parents=['railall'])
#         warp31 = Node('warp31', data=np.array([[0.98, 0.02],  # P(False|railall=False), P(False|railall=True)
#                                               [0.02, 0.98]]),  # P(True|railall=False), P(True|railall=True)
#                       parents=['railall'])
#         gage = Node('gage', data=np.array([[0.85, 0.15],  # P(False|railall=False), P(False|railall=True)
#                                           [0.15, 0.85]]),  # P(True|railall=False), P(True|railall=True)
#                     parents=['railall'])
#         cant = Node('cant', data=np.array([[0.63, 0.37],  # P(False|railall=False), P(False|railall=True)
#                                           [0.37, 0.63]]),  # P(True|railall=False), P(True|railall=True)
#                     parents=['railall'])
#         bn = Graph({'railall': railall, 'alignment': alignment, 'crosslevel': crosslevel,
#                     'warp62': warp62, 'warp31': warp31, 'gage': gage, 'cant': cant})
#         return bn

#     def compute_log_likelihood(self, samples, evidence=None):
#         log_lik = np.zeros((samples.shape[0],))
#         if evidence is None:
#             return log_lik  # Default to uniform likelihood if no evidence
#         node_name = list(evidence.keys())[0]  # Assuming single evidence for now
#         node_idx = list(self.graph.nodes.keys()).index(node_name)
#         node_val = evidence[node_name]
#         for i in range(samples.shape[0]):
#             r = samples[i, 0]  # RailAll value
#             log_lik[i] = np.log(max([1e-3, self.graph.nodes[node_name].data[node_val, r]]))
#         return log_lik

#     def compute_p_prior(self):
#         p_nodes = np.zeros((2, 2, 2, 2, 2, 2, 2))  # RailAll, Alignment, Crosslevel, Warp62, Warp31, Gage, Cant
#         prob = self.graph.nodes['railall'].data[0]
#         for a in range(2):
#             prob_a = prob * self.graph.nodes['alignment'].data[a, 0]
#             for c in range(2):
#                 prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, 0]
#                 for w62 in range(2):
#                     prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, 0]
#                     for w31 in range(2):
#                         prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, 0]
#                         for g in range(2):
#                             prob_g = prob_w31 * self.graph.nodes['gage'].data[g, 0]
#                             for ca in range(2):
#                                 p_nodes[0, a, c, w62, w31, g, ca] = prob_g * self.graph.nodes['cant'].data[ca, 0]
#         prob = self.graph.nodes['railall'].data[1]
#         for a in range(2):
#             prob_a = prob * self.graph.nodes['alignment'].data[a, 1]
#             for c in range(2):
#                 prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, 1]
#                 for w62 in range(2):
#                     prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, 1]
#                     for w31 in range(2):
#                         prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, 1]
#                         for g in range(2):
#                             prob_g = prob_w31 * self.graph.nodes['gage'].data[g, 1]
#                             for ca in range(2):
#                                 p_nodes[1, a, c, w62, w31, g, ca] = prob_g * self.graph.nodes['cant'].data[ca, 1]
#         return p_nodes / p_nodes.sum()

#     def compute_posterior(self, node_idx=0, node_val=1):
#         posterior = np.zeros((2, 2, 2, 2, 2, 2, 2))  # RailAll, Alignment, Crosslevel, Warp62, Warp31, Gage, Cant
#         total_prob = 0.0
#         for r in range(2):
#             prob = self.graph.nodes['railall'].data[r]
#             for a in range(2):
#                 prob_a = prob * self.graph.nodes['alignment'].data[a, r]
#                 for c in range(2):
#                     prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, r]
#                     for w62 in range(2):
#                         prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, r]
#                         for w31 in range(2):
#                             prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, r]
#                             for g in range(2):
#                                 prob_g = prob_w31 * self.graph.nodes['gage'].data[g, r]
#                                 for ca in range(2):
#                                     prob_ca = prob_g * self.graph.nodes['cant'].data[ca, r]
#                                     # Check evidence condition
#                                     evidence_match = False
#                                     node_name = list(self.graph.nodes.keys())[node_idx]
#                                     if node_name == 'alignment' and a == node_val:
#                                         evidence_match = True
#                                     elif node_name == 'crosslevel' and c == node_val:
#                                         evidence_match = True
#                                     elif node_name == 'warp62' and w62 == node_val:
#                                         evidence_match = True
#                                     elif node_name == 'warp31' and w31 == node_val:
#                                         evidence_match = True
#                                     elif node_name == 'gage' and g == node_val:
#                                         evidence_match = True
#                                     elif node_name == 'cant' and ca == node_val:
#                                         evidence_match = True
#                                     if evidence_match:
#                                         posterior[r, a, c, w62, w31, g, ca] = prob_ca
#                                         total_prob += prob_ca
#         # Normalize, avoiding division by zero
#         if total_prob > 0:
#             posterior /= total_prob
#         else:
#             posterior.fill(1.0 / posterior.size)  # Uniform distribution if no valid probability
#         return posterior

#     def compute_tvd(self, samples):
#         posterior = self.compute_posterior()
#         q_bm = self.born_machine.q_bm if hasattr(self.born_machine, 'q_bm') else np.zeros_like(posterior)
#         return np.abs(posterior - q_bm).max()


# class QueryRailAll(Query):

#     def __init__(self):
#         super().__init__()
#         self.graph_orig = RailAllBN.create_graph()


# class QBNQueryRailAll(QBNQuery):

#     def __init__(self):
#         super().__init__()
#         self.graph_orig = RailAllBN.create_graph()
#         self.rebuild_qbn()


# def main():
#     QS = QBNQueryRailAll()
#     n_shots = 8000

#     result = QS.qbn.perform_sampling(shots=n_shots)
#     P_RailAll1 = 0
#     for k, v in result.items():
#         if k[0] == '1':
#             P_RailAll1 += v
#     P_RailAll1 = P_RailAll1 / n_shots

#     P_A1R1 = 0
#     for k, v in result.items():
#         if k[0] == '1' and k[1] == '1':
#             P_A1R1 += v
#     P_A1R1 = P_A1R1 / n_shots
#     QS.target = {'alignment': 'alignment1'}
#     QS.evidence = {'railall': 'railall1'}
#     exact_A1_R1 = QS.get_true_result()
#     print(f'P(Alignment=1|RailAll=1)={P_A1R1 / P_RailAll1:.3f}, true={exact_A1_R1:.3f}')

#     return


# if __name__ == "__main__":
#     main()
################################
import numpy as np
from qubayes.qubayes.qubayes_tools import Query, QBNQuery, Node, Graph, BayesNet


class RailAllBN(BayesNet):

    def __init__(self, random_cpd=True):
        self.random_cpd = random_cpd
        self.graph = self.create_graph()
        if random_cpd:
            self.graph.nodes['railall'].data[0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['railall'].data[1] = 1. - self.graph.nodes['railall'].data[0]
            for node in ['alignment', 'crosslevel', 'warp62', 'warp31', 'gage', 'cant']:
                self.graph.nodes[node].data[0, 0] = np.random.uniform(low=0.01, high=0.99)
                self.graph.nodes[node].data[1, 0] = 1. - self.graph.nodes[node].data[0, 0]
                self.graph.nodes[node].data[0, 1] = np.random.uniform(low=0.01, high=0.99)
                self.graph.nodes[node].data[1, 1] = 1. - self.graph.nodes[node].data[0, 1]

    @staticmethod
    def create_graph():
        railall = Node('railall', data=np.array([0.18, 0.82]))  # True: 0.18, False: 0.82
        alignment = Node('alignment', data=np.array([[0.946, 0.054],  # P(False|railall=False), P(False|railall=True)
                                                    [0.054, 0.946]]),  # P(True|railall=False), P(True|railall=True)
                         parents=['railall'])
        crosslevel = Node('crosslevel', data=np.array([[0.70, 0.30],  # P(False|railall=False), P(False|railall=True)
                                                      [0.30, 0.70]]),  # P(True|railall=False), P(True|railall=True)
                          parents=['railall'])
        warp62 = Node('warp62', data=np.array([[0.905, 0.095],  # P(False|railall=False), P(False|railall=True)
                                              [0.095, 0.905]]),  # P(True|railall=False), P(True|railall=True)
                      parents=['railall'])
        warp31 = Node('warp31', data=np.array([[0.98, 0.02],  # P(False|railall=False), P(False|railall=True)
                                              [0.02, 0.98]]),  # P(True|railall=False), P(True|railall=True)
                      parents=['railall'])
        gage = Node('gage', data=np.array([[0.85, 0.15],  # P(False|railall=False), P(False|railall=True)
                                          [0.15, 0.85]]),  # P(True|railall=False), P(True|railall=True)
                    parents=['railall'])
        cant = Node('cant', data=np.array([[0.63, 0.37],  # P(False|railall=False), P(False|railall=True)
                                          [0.37, 0.63]]),  # P(True|railall=False), P(True|railall=True)
                    parents=['railall'])
        bn = Graph({'railall': railall, 'alignment': alignment, 'crosslevel': crosslevel,
                    'warp62': warp62, 'warp31': warp31, 'gage': gage, 'cant': cant})
        return bn

    def compute_log_likelihood(self, samples, evidence={"alignment": 0}):
        log_lik = np.zeros((samples.shape[0],))
        if evidence is None:
            return log_lik  # Default to uniform likelihood if no evidence
        node_name = list(evidence.keys())[0]  # Assuming single evidence for now
        node_idx = list(self.graph.nodes.keys()).index(node_name)
        node_val = evidence[node_name]
        for i in range(samples.shape[0]):
            r = samples[i, 0]  # RailAll value
            log_lik[i] = np.log(max([1e-3, self.graph.nodes[node_name].data[node_val, r]]))
        return log_lik

    def compute_p_prior(self):
        p_nodes = np.zeros((2, 2, 2, 2, 2, 2, 2))  # RailAll, Alignment, Crosslevel, Warp62, Warp31, Gage, Cant
        prob = self.graph.nodes['railall'].data[0]
        for a in range(2):
            prob_a = prob * self.graph.nodes['alignment'].data[a, 0]
            for c in range(2):
                prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, 0]
                for w62 in range(2):
                    prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, 0]
                    for w31 in range(2):
                        prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, 0]
                        for g in range(2):
                            prob_g = prob_w31 * self.graph.nodes['gage'].data[g, 0]
                            for ca in range(2):
                                p_nodes[0, a, c, w62, w31, g, ca] = prob_g * self.graph.nodes['cant'].data[ca, 0]
        prob = self.graph.nodes['railall'].data[1]
        for a in range(2):
            prob_a = prob * self.graph.nodes['alignment'].data[a, 1]
            for c in range(2):
                prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, 1]
                for w62 in range(2):
                    prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, 1]
                    for w31 in range(2):
                        prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, 1]
                        for g in range(2):
                            prob_g = prob_w31 * self.graph.nodes['gage'].data[g, 1]
                            for ca in range(2):
                                p_nodes[1, a, c, w62, w31, g, ca] = prob_g * self.graph.nodes['cant'].data[ca, 1]
        return p_nodes / p_nodes.sum()
    
    # def compute_posterior(self, node_idx=0, node_val=1):
    #     posterior = np.zeros((2, 2, 2, 2, 2, 2, 2))  # RailAll, Alignment, Crosslevel, Warp62, Warp31, Gage, Cant
    #     total_prob = 0.0
    #     for r in range(2):
    #         prob = self.graph.nodes['railall'].data[r]
    #         for a in range(2):
    #             if list(self.graph.nodes.keys())[node_idx] == 'alignment' and a != node_val:
    #                 continue  # Skip states where alignment doesn't match evidence
    #             prob_a = prob * self.graph.nodes['alignment'].data[a, r]
    #             # Precompute marginal for A=1
    #             if a == node_val and list(self.graph.nodes.keys())[node_idx] == 'alignment':
    #                 total_prob += prob_a * (self.graph.nodes['crosslevel'].data[0, r] + self.graph.nodes['crosslevel'].data[1, r]) * \
    #                               (self.graph.nodes['warp62'].data[0, r] + self.graph.nodes['warp62'].data[1, r]) * \
    #                               (self.graph.nodes['warp31'].data[0, r] + self.graph.nodes['warp31'].data[1, r]) * \
    #                               (self.graph.nodes['gage'].data[0, r] + self.graph.nodes['gage'].data[1, r]) * \
    #                               (self.graph.nodes['cant'].data[0, r] + self.graph.nodes['cant'].data[1, r])
    #             for c in range(2):
    #                 prob_c = prob_a * self.graph.nodes['crosslevel'].data[c, r]
    #                 for w62 in range(2):
    #                     prob_w62 = prob_c * self.graph.nodes['warp62'].data[w62, r]
    #                     for w31 in range(2):
    #                         prob_w31 = prob_w62 * self.graph.nodes['warp31'].data[w31, r]
    #                         for g in range(2):
    #                             prob_g = prob_w31 * self.graph.nodes['gage'].data[g, r]
    #                             for ca in range(2):
    #                                 prob_ca = prob_g * self.graph.nodes['cant'].data[ca, r]
    #                                 posterior[r, a, c, w62, w31, g, ca] = prob_ca
    #     if total_prob > 0:
    #         posterior /= total_prob
    #     else:
    #         posterior.fill(1.0 / posterior.size)  # Uniform if no valid probability
    #     print(f"total_prob: {total_prob}, posterior[0,1,0,0,0,0,0]: {posterior[0,1,0,0,0,0,0]}")
    #     return posterior
    # def compute_posterior(self, evidence={"alignment": 0}):
    #     # , "crosslevel": 0, "warp62": 0, "warp31": 0, "gage": 0, "cant": 0
    #     nodes = list(self.graph.nodes.keys())
    #     sz = [2]*len(nodes)
    #     posterior = np.zeros(sz)
    #     total_prob = 0.0
    #     for r in range(2):
    #         assignment = {"railall": r}
    #         match = True
    #         for k,v in evidence.items():
    #             assignment[k] = v
    #         prob = (
    #             self.graph.nodes['railall'].data[r] *
    #             self.graph.nodes['alignment'].data[evidence['alignment'], r] *
    #             self.graph.nodes['crosslevel'].data[evidence['crosslevel'], r] *
    #             self.graph.nodes['warp62'].data[evidence['warp62'], r] *
    #             self.graph.nodes['warp31'].data[evidence['warp31'], r] *
    #             self.graph.nodes['gage'].data[evidence['gage'], r] *
    #             self.graph.nodes['cant'].data[evidence['cant'], r]
    #         )
    #         # Posterior index order is [r,a,c,w62,w31,g,ca]!
    #         idx = (r, evidence['alignment'], evidence['crosslevel'], evidence['warp62'],
    #                evidence['warp31'], evidence['gage'], evidence['cant'])
    #         posterior[idx] = prob
    #         total_prob += prob
    #     if total_prob > 0:
    #         posterior /= total_prob
    #     print(f"Posterior[0,1,0,0,0,0,0]={posterior[0,0,0,0,0,0,0]:.6f}")
    #     print(f"Posterior[1,1,0,0,0,0,0]={posterior[1,0,0,0,0,0,0]:.6f}")
    #     print(f"Sum: {posterior.sum()}")
    #     return posterior

    def compute_posterior(self, evidence={"alignment": 0}):
        """
        Compute the full posterior joint over all variables,
        given evidence={'alignment': 1} (i.e., only alignment is fixed).
        Returns a tensor of shape [railall, alignment, crosslevel, warp62, warp31, gage, cant]
        """
        # Map node names to consistent order
        nodes = ['railall', 'alignment', 'crosslevel', 'warp62', 'warp31', 'gage', 'cant']
        sz = [2] * len(nodes)
        posterior = np.zeros(sz)
        total_prob = 0.0
    
        # Loop over all possible assignments
        for r in range(2):
            for a in range(2):
                for c in range(2):
                    for w62 in range(2):
                        for w31 in range(2):
                            for g in range(2):
                                for ca in range(2):
                                    if a != evidence['alignment']:
                                        continue  # only include assignments with alignment=1
                                    prob = (
                                        self.graph.nodes['railall'].data[r] *
                                        self.graph.nodes['alignment'].data[a, r] *
                                        self.graph.nodes['crosslevel'].data[c, r] *
                                        self.graph.nodes['warp62'].data[w62, r] *
                                        self.graph.nodes['warp31'].data[w31, r] *
                                        self.graph.nodes['gage'].data[g, r] *
                                        self.graph.nodes['cant'].data[ca, r]
                                    )
                                    posterior[r, a, c, w62, w31, g, ca] = prob
                                    total_prob += prob
        if total_prob > 0:
            posterior /= total_prob
        else:
            posterior.fill(1. / posterior.size)
        # print("Sum of all posterior entries =", posterior.sum())
        return posterior



    def compute_tvd(self, samples):
        posterior = self.compute_posterior()
        q_bm = self.born_machine.q_bm if hasattr(self.born_machine, 'q_bm') else np.zeros_like(posterior)
        return np.abs(posterior - q_bm).max()


class QueryRailAll(Query):

    def __init__(self):
        super().__init__()
        self.graph_orig = RailAllBN.create_graph()


class QBNQueryRailAll(QBNQuery):

    def __init__(self):
        super().__init__()
        self.graph_orig = RailAllBN.create_graph()
        self.rebuild_qbn()


def main():
    QS = QBNQueryRailAll()
    n_shots = 8000

    result = QS.qbn.perform_sampling(shots=n_shots)
    P_RailAll1 = 0
    for k, v in result.items():
        if k[0] == '1':
            P_RailAll1 += v
    P_RailAll1 = P_RailAll1 / n_shots

    P_A1R1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[1] == '1':
            P_A1R1 += v
    P_A1R1 = P_A1R1 / n_shots
    QS.target = {'alignment': 'alignment1'}
    QS.evidence = {'railall': 'railall1'}
    exact_A1_R1 = QS.get_true_result()
    print(f'P(Alignment=1|RailAll=1)={P_A1R1 / P_RailAll1:.3f}, true={exact_A1_R1:.3f}')

    return


if __name__ == "__main__":
    main()