from qubayes.cyber_attack_bn import CyberAttackA2BN
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy
import matplotlib.pyplot as plt
from qubayes.qubayes_tools import Node, Graph, BayesNet
import itertools
from qiskit_algorithms import optimizers


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def counts_to_cpd(counts, reverse=True):
    n_dim     = len(list(counts.keys())[0])
    n_samples = sum(counts.values())
    lst = list(itertools.product([0, 1], repeat=n_dim))
    cpd = np.zeros((2,) * n_dim, dtype=float)
    for c in lst:
        idx = c[::-1] if reverse else c
        key = "".join([str(x) for x in idx])
        if key in counts:
            cpd[c] = float(counts[key]) / n_samples
    return cpd


class OptimalClassifier:

    def __init__(self, bayes_net):
        self.bn          = bayes_net
        self.q_posterior = None
        self.p_prior     = bayes_net.compute_p_prior()

    def train(self, train_x, train_y, learning_rate=None):
        train_x = train_x[train_y == 0, :]
        unique_rows, unique_counts = np.unique(train_x, axis=0, return_counts=True)
        lst        = list(itertools.product([0, 1], repeat=self.bn.graph.n_variables - 1))
        estimation = np.zeros((2,) * (self.bn.graph.n_variables - 1), dtype=float)
        for c in lst:
            idx = (unique_rows == np.array([c])).all(axis=1)
            if idx.any():
                estimation[c] = float(unique_counts[idx][0]) / train_x.shape[0]
        self.q_posterior = estimation

    def predict(self, samples, labels=None):
        if labels is None:
            labels = np.zeros(samples.shape[0])
        self.train(samples, labels)
        pred = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            idx     = tuple(samples[i, :])
            pred[i] = self.q_posterior[idx] / (self.q_posterior[idx] + self.p_prior[idx])
        return pred

    def compute_loss(self, train_x, train_y):
        p_prior    = self.predict(train_x, labels=train_y)
        E_log_bm   = np.log(1 - p_prior[train_y == 0]).mean()
        E_log_prior = np.log(p_prior[train_y == 1]).mean()
        return E_log_bm + E_log_prior


class MLP_Classifier:

    def __init__(self, n_inputs):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(6, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])

    def train(self, train_x, train_y, learning_rate=0.03):
        idx     = np.random.permutation(train_x.shape[0])
        train_x = train_x[idx, :]
        train_y = train_y[idx]
        split   = int(train_x.shape[0] * 0.2)
        val_x, val_y       = train_x[:split, :], train_y[:split]
        train_x, train_y   = train_x[split:, :], train_y[split:]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'],
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = self.model.fit(
            train_x, train_y,
            validation_data=(val_x, val_y),
            epochs=200, batch_size=10,
            verbose=0, callbacks=[early_stop],
        )
        best_epoch = early_stop.stopped_epoch - early_stop.patience + 1
        print(f"- Validation accuracy {history.history['val_accuracy'][best_epoch]:.2f} "
              f"at epoch {best_epoch}.")
        return history

    def compute_loss(self, train_x, train_y):
        p_prior    = self.predict(train_x)
        E_log_bm   = np.log(1 - p_prior[train_y == 0]).mean()
        E_log_prior = np.log(p_prior[train_y == 1]).mean()
        return E_log_bm + E_log_prior

    def predict(self, samples):
        return self.model.predict(samples, verbose=0)[:, 0]


class Optimizer:
    """Gradient-based optimizer using the parameter-shift rule."""

    def __init__(self, born_machine, bayes_net, classifier,
                 n_iterations=100, learning_rate=0.003):
        self.n_iterations  = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net     = bayes_net
        self.classifier    = classifier
        self.born_machine  = born_machine

    def estimate_gradient(self, n_samples=100):
        shift     = np.pi / 2
        gradients = np.zeros(self.born_machine.params.shape)
        for i in range(len(self.born_machine.params)):
            bm = {k: deepcopy(self.born_machine) for k in ('plus', 'minus')}
            bm['plus'].params[i]  += shift
            bm['minus'].params[i] -= shift
            md = {}
            for key in ('plus', 'minus'):
                samples_crs = bm[key].sample(n_samples)
                p_prior     = self.classifier.predict(samples_crs)
                loglik      = self.bayes_net.compute_log_likelihood(samples_crs)
                md[key]     = (logit(1. - p_prior) - loglik).mean()
            gradients[i] = (md['plus'] - md['minus']) / 2
        return gradients

    def compute_kl_loss(self, samples_crs=None, evidence=None):
        if samples_crs is None:
            samples_crs = self.born_machine.sample(100)
        p_prior = self.classifier.predict(samples_crs)
        loglik  = self.bayes_net.compute_log_likelihood(samples_crs, evidence)
        return (logit(1. - p_prior) - loglik).mean()

    def optimize(self):
        metrics = {k: np.zeros(self.n_iterations) for k in ('tvd', 'kl_loss', 'ce_loss')}
        for i in range(self.n_iterations):
            s_bm    = self.born_machine.sample(100)
            s_prior = self.bayes_net.sample_from_prior(100)
            x_train = np.vstack((s_bm, s_prior))
            y_train = np.zeros(x_train.shape[0])
            y_train[s_bm.shape[0]:] = 1
            self.classifier.train(x_train, y_train)
            self.born_machine.params += self.learning_rate * self.estimate_gradient()
            metrics['kl_loss'][i] = self.compute_kl_loss(s_bm)
            metrics['tvd'][i]     = self.compute_tvd(s_bm)
            metrics['ce_loss'][i] = self.classifier.compute_loss(x_train, y_train)
            print(f"Iteration {i+1}: tvd={metrics['tvd'][i]:.4f}, "
                  f"loss={metrics['kl_loss'][i]:.4f}, clf={metrics['ce_loss'][i]:.4f}")
        return self.born_machine, metrics

    def compute_tvd(self, samples):
        return self.bayes_net.compute_tvd(samples)


class DerivativeFreeOptimizer:

    def __init__(self, born_machine, bayes_net, classifier,
                 evidence=None, n_iterations=100,
                 learning_rate=0.003, method='COBYLA'):
        self.n_iterations  = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net     = bayes_net
        self.classifier    = classifier
        self.born_machine  = born_machine
        self.evidence      = evidence or {}
        self.method        = method

    def compute_kl_loss(self, theta_values):
        n_samples = 1000
        self.born_machine.params = theta_values
        samples = self.born_machine.sample(n_samples, return_samples=True)
        q_bm    = self.born_machine.q_bm
        if q_bm.ndim == 1:
            q_bm = q_bm.reshape(2, 2)

        p_prior = self.bayes_net.compute_p_prior()
        if p_prior.ndim == 1:
            p_prior = p_prior.reshape(2, 2)

        logliks = self.bayes_net.compute_log_likelihood(samples, evidence=self.evidence)

        loss = 0.0
        for i in range(samples.shape[0]):
            q_s = float(q_bm[tuple(samples[i])])
            p_s = float(p_prior[tuple(samples[i])])
            if q_s <= 0.0 or p_s <= 0.0:
                continue
            loss += np.log(q_s) - np.log(p_s) - logliks[i]
        return float(loss) / n_samples

    def optimize(self):
        optimizer  = getattr(optimizers, self.method)
        info       = {'parameters': [], 'loss': []}

        if self.method == 'COBYLA':
            callback = lambda params: info['parameters'].append(params)
        elif self.method == 'GradientDescent':
            def callback(nfev, params, loss, grad_norm):
                info['parameters'].append(params)
                info['loss'].append(loss)

        opt        = optimizer(maxiter=self.n_iterations, callback=callback, tol=1e-10)
        opt.minimize(self.compute_kl_loss, self.born_machine.params)

        n_iters = len(info['parameters'])
        metrics = {k: np.zeros(n_iters) for k in ('tvd', 'kl_loss', 'ce_loss')}

        for i, params in enumerate(info['parameters']):
            metrics['kl_loss'][i]  = self.compute_kl_loss(params)
            self.born_machine.params = params
            self.born_machine.sample(1000, return_samples=False)
            pred      = float(self.born_machine.q_bm[1, 1])
            posterior = self.bayes_net.compute_posterior(evidence=self.evidence)
            metrics['tvd'][i] = self.compute_tvd()
            print(f"Iteration {i+1}: Loss={metrics['kl_loss'][i]:.4f}, "
                  f"Pred={pred:.3f}, True={float(posterior[1, 1]):.6f}")

        return self.born_machine, metrics

    def compute_tvd(self):
        posterior = self.bayes_net.compute_posterior(evidence=self.evidence)
        return float(np.abs(posterior - self.born_machine.q_bm).max())


class BornMachine:

    def __init__(self, n_qubits, n_blocks=0, ansatz_type='RealAmplitudes'):
        self.n_qubits    = n_qubits
        self.n_blocks    = n_blocks
        self.ansatz_type = ansatz_type
        self.params      = None
        self.ansatz      = None
        self.q_bm        = None
        self._init_ansatz()

    def _init_ansatz(self):
        if self.ansatz_type == 'RealAmplitudes':
            self.ansatz = RealAmplitudes(self.n_qubits, reps=self.n_blocks,
                                         entanglement='linear')
        else:
            self.ansatz = EfficientSU2(self.n_qubits, su2_gates=['rz', 'rx'],
                                       reps=self.n_blocks, entanglement='linear')
        if self.params is None:
            self.params = np.random.normal(0, 0.1, size=self.ansatz.num_parameters)

    def reset_ansatz(self):
        self._init_ansatz()
        param_dict = dict(zip(self.ansatz.parameters, self.params))
        self.ansatz.assign_parameters(param_dict, inplace=True)

    def print_circuit(self):
        print(self.ansatz.decompose())

    def sample(self, n_samples, return_samples=True):
        self.reset_ansatz()
        circuit = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            circuit.h(q)
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()

        simulator = AerSimulator(method='matrix_product_state')
        result    = simulator.run(
            transpile(circuit, simulator), shots=n_samples, memory=True
        ).result()

        counts    = result.get_counts()
        self.q_bm = counts_to_cpd(counts, reverse=True)

        if return_samples:
            raw     = result.get_memory()
            samples = np.array(
                [[c == '1' for c in s[::-1]] for s in raw], dtype='int32'
            )
            return samples
        return self.q_bm


class BNStructuredBornMachine:
    """
    Problem-inspired 2-qubit Born machine that mirrors the QBN circuit encoding.

    Instead of the generic RY+CNOT pattern (RealAmplitudes), this circuit uses
    conditional RY gates that directly mirror how the QBN encodes a 4-state CPT:

        q0: ─ RY(θ₀) ─ X ─ ●──── X ─ ●──── measure
        q1: ──────────────── RY(θ₁) ── RY(θ₂) ── measure

    where:
        P(b0 = 1)        = sin²(θ₀/2)
        P(b1 = 1 | b0=0) = sin²(θ₁/2)   [CRY fires after X flip when b0=0→q0=1]
        P(b1 = 1 | b0=1) = sin²(θ₂/2)   [CRY fires when q0=1, i.e. b0=1]

    State index = b0 + 2·b1 (same encoding as CyberAttackA2BN / CyberAttackA1BN).

    3 parameters vs RealAmplitudes-reps1 (4 params).  Each parameter has a direct
    probabilistic interpretation.  No H-gate initialisation — circuit starts from |00⟩.
    """

    def __init__(self, n_qubits=2):
        assert n_qubits == 2, "BNStructuredBornMachine requires exactly 2 qubits."
        self.n_qubits    = n_qubits
        self.n_blocks    = 0           # informational only
        self.ansatz_type = 'BNStructured'
        self.q_bm        = None
        # Initialise angles uniformly in [0, π] so probabilities span [0, 1]
        self.params = np.random.uniform(0.1, np.pi - 0.1, size=3)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the BN-structured 2-qubit circuit with current params."""
        θ = self.params
        qc = QuantumCircuit(self.n_qubits)
        qc.ry(float(θ[0]), 0)      # encodes P(b0=1)
        qc.x(0)                     # flip q0 so CRY below fires on original b0=0
        qc.cry(float(θ[1]), 0, 1)  # encodes P(b1=1 | b0=0)
        qc.x(0)                     # restore q0
        qc.cry(float(θ[2]), 0, 1)  # encodes P(b1=1 | b0=1)
        return qc

    def sample(self, n_samples, return_samples=True):
        circuit = self._build_circuit()
        circuit.measure_all()
        simulator = AerSimulator(method='matrix_product_state')
        result    = simulator.run(
            transpile(circuit, simulator), shots=n_samples, memory=True
        ).result()
        counts    = result.get_counts()
        self.q_bm = counts_to_cpd(counts, reverse=True)
        if return_samples:
            raw     = result.get_memory()
            samples = np.array(
                [[c == '1' for c in s[::-1]] for s in raw], dtype='int32'
            )
            return samples
        return self.q_bm

    def reset_ansatz(self):
        pass   # not needed; circuit is rebuilt from params in every sample() call

    def print_circuit(self):
        print(self._build_circuit().decompose())


def plot_optimization_metrics(metrics, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(metrics['kl_loss'], label='Loss (Eq. 7)')
    axes[0].set_xlabel('Epoch');  axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(metrics['tvd'], label='TVD: q(z|x) vs p(z|x)')
    if metrics['ce_loss'].max() > metrics['ce_loss'].min():
        axes[1].plot(metrics['ce_loss'], label='Classifier loss')
    axes[1].set_xlabel('Epoch');  axes[1].set_ylabel('TVD')
    axes[1].legend()
    if save_path:
        plt.savefig(save_path)
        print(f'Saved to {save_path}')
    else:
        plt.show()


def main():
    bn  = CyberAttackA2BN()
    bm  = BornMachine(n_qubits=2, n_blocks=1, ansatz_type='RealAmplitudes')
    clf = OptimalClassifier(bn)

    evidence = {
        "FlowState_2":  1,   # Abnormal
        "ValveState_2": 1,   # Abnormal
    }

    opt = DerivativeFreeOptimizer(
        born_machine=bm,
        bayes_net=bn,
        classifier=clf,
        evidence=evidence,
        n_iterations=400,
        method='COBYLA',
    )
    bm_opt, metrics = opt.optimize()
    plot_optimization_metrics(metrics)

    posterior = bn.compute_posterior(evidence=evidence)
    print("\nExact posterior over (b0,b1):")
    print(posterior)
    if bm_opt.q_bm is not None:
        print("Learned q_bm over (b0,b1):")
        print(bm_opt.q_bm)


if __name__ == "__main__":
    main()
