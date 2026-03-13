from qubayes.qubayes.qubayes_tools import Node, Graph, BayesNet
import numpy as np

class RailDefectBN(BayesNet):
    @staticmethod
    def create_graph():
        bfi = Node("BFI", data=np.array([0.5812, 0.3307, 0.0766, 0.0115]), states={"Clean": 0, "Moderately Clean": 1, "Moderately Fouled": 2, "Fouled+": 3})
        bti = Node("BTI", data=np.array([0.048, 0.3046, 0.4843, 0.1631]), states={"Thick": 0, "Design": 1, "Reduced": 2, "Thin+": 3})
        lri = Node("LRI", data=np.array([0.5858, 0.4142]), states={"Good": 0, "Not Good": 1})
        mli = Node("MLI", data=np.array([0.8054, 0.1946]), states={"Good": 0, "Not Good": 1})

        bmh_data = np.zeros((2, 4, 2))
        for i_bfi in range(4):
            for i_mli in range(2):
                score = 0.75 * i_bfi / 3 + 0.25 * i_mli
                bmh_data[0, i_bfi, i_mli] = 1 - score
                bmh_data[1, i_bfi, i_mli] = score
        bmh = Node("BMH", parents=["BFI", "MLI"], data=bmh_data, states={"Good": 0, "Poor": 1})

        bsh_data = np.zeros((2, 4, 2))
        for i_bti in range(4):
            for i_lri in range(2):
                score = 0.5 * (3 - i_bti) / 3 + 0.5 * i_lri
                bsh_data[0, i_bti, i_lri] = 1 - score
                bsh_data[1, i_bti, i_lri] = score
        bsh = Node("BSH", parents=["BTI", "LRI"], data=bsh_data, states={"Good": 0, "Poor": 1})

        defect_data = np.zeros((2, 2, 2))
        for bmh_bin in range(2):
            for bsh_bin in range(2):
                score = 0.5 * bmh_bin + 0.5 * bsh_bin
                defect_data[0, bmh_bin, bsh_bin] = 1 - score
                defect_data[1, bmh_bin, bsh_bin] = score
        defect = Node("Defect", parents=["BMH", "BSH"], data=defect_data, states={"Absent": 0, "Present": 1})

        graph = Graph({"BFI": bfi, "BTI": bti, "LRI": lri, "MLI": mli, "BMH": bmh, "BSH": bsh, "Defect": defect})
        return graph

    def __init__(self):
        super().__init__(self.create_graph())

    def compute_p_prior(self):
        return self.graph.marginalize_all_but(["BFI", "BTI", "LRI", "MLI"])

    def compute_log_likelihood(self, samples, defect=1):
        log_lik = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            bfi, bti, lri, mli = samples[i]
            bmh_probs = self.graph.nodes["BMH"].data[:, bfi, mli]
            bsh_probs = self.graph.nodes["BSH"].data[:, bti, lri]
            for bmh_val in range(2):
                for bsh_val in range(2):
                    p_bmh = bmh_probs[bmh_val]
                    p_bsh = bsh_probs[bsh_val]
                    p_defect = self.graph.nodes["Defect"].data[defect, bmh_val, bsh_val]
                    log_lik[i] += p_bmh * p_bsh * p_defect
            log_lik[i] = np.log(max(log_lik[i], 1e-5))
        return log_lik
