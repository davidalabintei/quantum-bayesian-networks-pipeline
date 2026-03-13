
#############################################################################
# from qubayes.qubayes.rail_all_bn import RailAllBN
from qubayes.qubayes.cyber_attack_bn import CyberAttackA2BN
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy
import matplotlib.pyplot as plt
from qubayes.qubayes.qubayes_tools import Node, Graph, BayesNet
import itertools
from qiskit_algorithms import optimizers

def logit(x):
    eps = 1e-5
    x = np.clip(x, eps, 1 - eps)  # Clip values to avoid log(0) or log(inf)
    return np.log(x / (1 - x))

def counts_to_cpd(counts, reverse=True):
    n_dim = len(list(counts.keys())[0])
    n_samples = sum(counts.values())
    lst = list(itertools.product([0, 1], repeat=n_dim))
    cpd = np.zeros((2,) * n_dim, dtype=float)
    states = np.array(list(counts.keys()))
    for c in lst:
        if reverse:
            idx = c[::-1]
        else:
            idx = c
        key = "".join([str(x) for x in idx])
        if key in list(counts.keys()):
            cpd[c] = float(counts[key]) / n_samples
    return cpd


class OptimalClassifier(object):

    def __init__(self, bayes_net):
        self.bn = bayes_net
        self.q_posterior = None
        self.p_prior = bayes_net.compute_p_prior()

    def train(self, train_x, train_y, learning_rate=None):
        train_x = train_x[train_y == 0, :]
        unique_rows, unique_counts = np.unique(train_x, axis=0, return_counts=True)
        lst = list(itertools.product([0, 1], repeat=self.bn.graph.n_variables-1))
        estimation = np.zeros((2,) * (self.bn.graph.n_variables-1), dtype=float)
        for c in lst:
            idx = (unique_rows == np.array([c])).all(axis=1)
            if idx.any():
                estimation[c] = float(unique_counts[idx][0]) / train_x.shape[0]
        self.q_posterior = estimation
        return None

    def predict(self, samples, labels=None):
        if labels is None:
            labels = np.zeros((samples.shape[0],))
        self.train(samples, labels)
        pred = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            idx = tuple(samples[i, :])
            pred[i] = self.q_posterior[idx] / (self.q_posterior[idx] + self.p_prior[idx])
        return pred

    def compute_loss(self, train_x, train_y):
        p_prior = self.predict(train_x, labels=train_y)
        E_log_bm = (np.log(1 - p_prior[train_y == 0])).mean()
        E_log_prior = (np.log(p_prior[train_y == 1])).mean()
        return E_log_bm + E_log_prior


class MLP_Classifier(object):

    def __init__(self, n_inputs):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(6, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def train(self, train_x, train_y, learning_rate=0.03):
        idx = np.random.permutation(train_x.shape[0])
        train_x = train_x[idx, :]
        train_y = train_y[idx]
        split = 0.2
        idx = int(train_x.shape[0] * split)
        val_x = train_x[:idx, :]
        val_y = train_y[:idx]
        train_x = train_x[idx:, :]
        train_y = train_y[idx:]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy'])
        history = self.model.fit(x=train_x, y=train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=200, batch_size=10,
                                 verbose=0, callbacks=[early_stopping])

        best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
        best_val_accuracy = history.history['val_accuracy'][best_epoch]
        print(f"- Validation accuracy {best_val_accuracy:.2f} at best epoch {best_epoch}.")

        return history

    def compute_loss(self, train_x, train_y):
        p_prior = self.predict(train_x)
        E_log_bm = (np.log(1 - p_prior[train_y == 0])).mean()
        E_log_prior = (np.log(p_prior[train_y == 1])).mean()
        return E_log_bm + E_log_prior

    def predict(self, samples):
        return self.model.predict(samples, verbose=0)[:, 0]


class Optimizer(object):

    def __init__(self, born_machine, bayes_net, classifier, n_iterations=100, learning_rate=0.003):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        self.classifier = classifier
        self.born_machine = born_machine

    def estimate_gradient(self, n_samples=100):
        shift = np.pi / 2
        gradients = np.zeros(self.born_machine.params.shape)

        for i in range(len(self.born_machine.params)):
            bm = {'plus': deepcopy(self.born_machine),
                  'minus': deepcopy(self.born_machine)}
            bm['plus'].params[i] += shift
            bm['minus'].params[i] -= shift
            md = dict()

            for key in ['plus', 'minus']:
                samples_crs = bm[key].sample(n_samples)
                p_prior = self.classifier.predict(samples_crs)
                p_bm = 1. - p_prior
                loglik = self.bayes_net.compute_log_likelihood(samples_crs)
                md[key] = (logit(p_bm) - loglik).mean()

            gradients[i] = (md['plus'] - md['minus']) / 2
        return gradients

    def compute_kl_loss(self, samples_crs=None, evidence=None):
        if samples_crs is None:
            samples_crs = self.born_machine.sample(100)
        p_prior = self.classifier.predict(samples_crs)
        p_bm = 1. - p_prior
        loglik = self.bayes_net.compute_log_likelihood(samples_crs, evidence)
        return (logit(p_bm) - loglik).mean()

    def optimize(self):
        metrics = {'tvd': np.zeros((self.n_iterations,)),
                   'kl_loss': np.zeros((self.n_iterations,)),
                   'ce_loss': np.zeros((self.n_iterations,))}
        for i in range(self.n_iterations):
            s_bm = self.born_machine.sample(100)
            s_prior = self.bayes_net.sample_from_prior(100)

            x_train = np.vstack((s_bm, s_prior))
            y_train = np.zeros((s_prior.shape[0] + s_bm.shape[0],))
            y_train[s_bm.shape[0]:] = 1
            history = self.classifier.train(x_train, y_train)

            gradients = self.estimate_gradient()

            self.born_machine.params += self.learning_rate * gradients

            metrics['kl_loss'][i] = self.compute_kl_loss(s_bm)
            metrics['tvd'][i] = self.compute_tvd(s_bm)
            metrics['ce_loss'][i] = self.classifier.compute_loss(x_train, y_train)

            print(f"Iteration {i + 1}: tvd = {metrics['tvd'][i]:.4f}, born loss = {metrics['kl_loss'][i]:.4f},"
                  f" clf loss = {metrics['ce_loss'][i]:.4f}")
        return self.born_machine, metrics

    def compute_tvd(self, samples):
        return self.bayes_net.compute_tvd(samples)


class DerivativeFreeOptimizer(object):

    def __init__(self, born_machine, bayes_net, classifier, n_iterations=100,
                 learning_rate=0.003, method='COBYLA'):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        self.classifier = classifier
        self.born_machine = born_machine
        self.method = method
        self.params = None

    # def compute_kl_loss(self, theta_values):
    #     n_samples = 1000
    #     self.born_machine.params = theta_values
    #     samples = self.born_machine.sample(n_samples, return_samples=True)  # 6D samples (child nodes)
    #     q_bm = self.born_machine.q_bm  # 6D CPD
    #     p_prior = self.bayes_net.compute_p_prior()  # 7D CPD
    #     # Marginalize p_prior over railall dimension (axis 0)
    #     p_prior_marginal = p_prior.sum(axis=0)  # Sum over railall (0/1), resulting in 6D
    #     # logliks = self.bayes_net.compute_log_likelihood(samples)
    #     evidence = getattr(self, "evidence", None)
    #     logliks = self.bayes_net.compute_log_likelihood(samples, evidence=evidence)

    #     loss = 0
    #     for i in range(samples.shape[0]):
    #         s = samples[i, :]  # 6-element array for child nodes
    #         q_bm_s = q_bm[tuple(s)]  # Scalar
    #         p_prior_s = p_prior_marginal[tuple(s)]  # Scalar (marginalized)
    #         if np.allclose(q_bm_s + p_prior_s, 0, atol=1e-10):  # Avoid division by zero
    #             d_bm = 0.5  # Default to 0.5 if sum is zero
    #         else:
    #             d_bm = q_bm_s / (q_bm_s + p_prior_s)  # Scalar division
    #         loss += logit(d_bm) - logliks[i]  # Scalar subtraction
    #     return loss / n_samples  # Scalar average
    def compute_kl_loss(self, theta_values):
        # match your existing style
        n_samples = 1000
    
        # 1) set BM params and sample so that self.born_machine.q_bm is populated
        self.born_machine.params = theta_values
        samples = self.born_machine.sample(n_samples, return_samples=True)  # shape (n_samples, 2) for cyber
        q_bm = self.born_machine.q_bm  # distribution tensor produced by counts_to_cpd
    
        # 2) ensure q_bm has shape (2,2) for 2 latent bits
        if q_bm.ndim == 1:
            q_bm = q_bm.reshape(2, 2)
    
        # 3) prior over same latent bits, must be (2,2) in the cyber BN wrapper
        p_prior = self.bayes_net.compute_p_prior()
        if p_prior.ndim == 1:
            p_prior = p_prior.reshape(2, 2)
    
        # 4) likelihood with evidence
        evidence = getattr(self, "evidence", None)
        logliks = self.bayes_net.compute_log_likelihood(samples, evidence=evidence)
    
        # 5) loss
        loss = 0.0
        for i in range(samples.shape[0]):
            s = samples[i, :]              # length 2
            q_s = float(q_bm[tuple(s)])
            p_s = float(p_prior[tuple(s)])
    
            if q_s <= 0.0 or p_s <= 0.0:
                continue
    
            loss += (np.log(q_s) - np.log(p_s) - logliks[i])
    
        return float(loss) / n_samples


    def optimize(self):
        optimizer = getattr(optimizers, self.method)
        info = {'parameters': [], 'loss': []}

        if self.method == 'COBYLA':
            def callback(parameters):
                info['parameters'].append(parameters)
        elif self.method == 'GradientDescent':
            def callback(nfev, parameters, loss_value, gradient_norm):
                info['parameters'].append(parameters)
                info['loss'].append(loss_value)
        opt = optimizer(maxiter=self.n_iterations, callback=callback, tol=1e-10)
        opt_results = opt.minimize(self.compute_kl_loss, self.born_machine.params)
        n_iterations = len(info['parameters'])
        metrics = {'tvd': np.zeros(n_iterations),
                   'kl_loss': np.zeros(n_iterations),
                   'ce_loss': np.zeros(n_iterations)}
        for i in range(len(info['parameters'])):
            metrics['kl_loss'][i] = self.compute_kl_loss(info['parameters'][i])
            self.born_machine.params = info['parameters'][i]
            pred = self.born_machine.sample(1000, return_samples=False)
            q_bm = self.born_machine.q_bm
            pred = float(q_bm[1, 1])

            metrics['tvd'][i] = self.compute_tvd()
            evidence = getattr(self, "evidence", None)
            posterior = self.bayes_net.compute_posterior(evidence=evidence)
            true_value = float(posterior[1, 1])

            # posterior = self.bayes_net.compute_posterior()
            # print(posterior)
            print(f"Iteration {i + 1}: Loss = {metrics['kl_loss'][i]:.4f}, "
                  # f"Pred = {pred.flat[0]:.6f}, "
                  f"Pred = {pred}, "
                  # f"True = {posterior.flat[0]:.6f}")
                  f"True = {true_value}")

        return self.born_machine, metrics

    def compute_tvd(self):
        # posterior = self.bayes_net.compute_posterior()
        evidence = getattr(self, "evidence", None)
        posterior = self.bayes_net.compute_posterior(evidence=evidence)

        tvd = (abs(posterior - self.born_machine.q_bm)).max()
        return tvd


class BornMachine(object):

    def __init__(self, n_qubits, n_blocks=0, ansatz_type='RealAmplitudes'):
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.params = None
        self.ansatz = None
        self.q_bm = None
        self.ansatz_type = ansatz_type
        self.reset_ansatz()

    def reset_ansatz(self):
        if self.ansatz_type == 'RealAmplitudes':
            self.ansatz = RealAmplitudes(self.n_qubits, reps=self.n_blocks,
                                         entanglement='linear')
        else:
            self.ansatz = EfficientSU2(self.n_qubits, su2_gates=['rz', 'rx'],
                                       reps=self.n_blocks,
                                       entanglement='linear')
        if self.params is None:
            self.params = np.random.normal(0, 0.1, size=self.ansatz.num_parameters)
        param_dict = {param: value for param, value in zip(self.ansatz.parameters, self.params)}
        self.ansatz.assign_parameters(param_dict, inplace=True)

    def print_circuit(self):
        print(self.ansatz.decompose())

    def sample(self, n_samples, return_samples=True):
        circuit = QuantumCircuit(self.n_qubits)
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        self.reset_ansatz()
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()

        simulator = AerSimulator(method='matrix_product_state')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=n_samples, memory=True).result()
        counts = result.get_counts()
        self.q_bm = counts_to_cpd(counts, reverse=True)
        if return_samples:
            samples = result.get_memory()
            out = np.array([[char == '1' for char in string[::-1]] for string in samples], dtype='int32')
            unique_rows, counts = np.unique(out, axis=0, return_counts=True)
            if len(counts) > 1:
                np.testing.assert_almost_equal(counts[1]/n_samples, self.q_bm[tuple(unique_rows[1])], decimal=3)
        else:
            out = self.q_bm
        return out


def plot_optimization_metrics(metrics, save=False):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(metrics['kl_loss'], label='Loss according to Eq. 7')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(metrics['tvd'], label='TVD between q(z|x) and p(z|x)')
    if metrics['ce_loss'].max() > metrics['ce_loss'].min():
        ax[1].plot(metrics['ce_loss'], label='Classifier loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('TVD')
    ax[1].legend()
    if save:
        out_fln = fr'C:\Users\krf\projects\Quanco\git\qubayes\figs\vi_classifier\optimization.png'
        plt.savefig(out_fln)
        print(f'Saved figure to {out_fln}')
    else:
        plt.show()


# def main():
#     bn = RailAllBN(random_cpd=False)

#     bm = BornMachine(len(bn.graph.nodes)-1, n_blocks=1,
#                      ansatz_type='RealAmplitudes')

#     classifier = OptimalClassifier(bn)

#     opt = DerivativeFreeOptimizer(bm, bn, classifier, n_iterations=400, learning_rate=0.003)
#     bm_opt, metrics = opt.optimize()
#     plot_optimization_metrics(metrics, save=0)
#     return
def main():
    # --- Build the cyber BN wrapper ---
    bn = CyberAttackA2BN()

    # Latent variable z is AttackType_2 encoded as 2 bits
    bm = BornMachine(n_qubits=2, n_blocks=1, ansatz_type='RealAmplitudes')

    classifier = OptimalClassifier(bn)

    # Evidence example that matches your rejection sampling setup
    evidence = {
        "FlowState_2": 1,   # Abnormal
        "ValveState_2": 1,  # Abnormal
        # "TankState_2": 1, # Optional
    }

    # Plug evidence into the optimizer through compute_log_likelihood calls
    # We add a tiny patch to DerivativeFreeOptimizer below to store evidence.
    opt = DerivativeFreeOptimizer(
        born_machine=bm,
        bayes_net=bn,
        classifier=classifier,
        n_iterations=400,
        learning_rate=0.003,
        method='COBYLA'
    )
    opt.evidence = evidence

    bm_opt, metrics = opt.optimize()
    plot_optimization_metrics(metrics, save=0)

    # Report final learned q(z|x) versus exact posterior p(z|x)
    posterior = bn.compute_posterior(evidence=evidence)
    print("Exact posterior over (b0,b1):")
    print(posterior)

    if bm_opt.q_bm is not None:
        print("Learned q_bm over (b0,b1):")
        print(bm_opt.q_bm)

    return


if __name__ == "__main__":
    main()