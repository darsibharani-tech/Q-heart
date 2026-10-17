"""Reproducible Q-HEART PS-Q1 benchmark.

The experiment intentionally keeps the learned model at four logical qubits.
Processor A/B are only physical coupling-map targets for compilation and noisy
evaluation. All results and plots are generated from this run.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROCESSORS = ROOT / "processors"
RESULTS.mkdir(exist_ok=True)
SEED = 42
N_QUBITS = 4
SHOTS = 4096
BASIS = ["rz", "sx", "x", "cx"]


def load_processor(name: str) -> dict:
    return json.loads((PROCESSORS / name).read_text(encoding="utf-8"))


def build_vqc() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    x = ParameterVector("x", N_QUBITS)
    theta = ParameterVector("theta", 2 * N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(x[q], q)
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)
    for q in range(N_QUBITS):
        qc.ry(theta[q], q)
    for q in range(N_QUBITS - 1, 0, -1):
        qc.cx(q, q - 1)
    for q in range(N_QUBITS):
        qc.rz(theta[N_QUBITS + q], q)
    return qc, x, theta


def bind_circuit(template, x_params, theta_params, features, weights, measure=False):
    qc = template.assign_parameters({**dict(zip(x_params, features)), **dict(zip(theta_params, weights))})
    if measure:
        measured = QuantumCircuit(N_QUBITS, N_QUBITS)
        measured.compose(qc, inplace=True)
        measured.measure(range(N_QUBITS), range(N_QUBITS))
        return measured
    return qc


def probability_one(features, weights, template, x_params, theta_params) -> float:
    state = Statevector.from_instruction(bind_circuit(template, x_params, theta_params, features, weights))
    observable = SparsePauliOp.from_list([(("I" * (N_QUBITS - 1)) + "Z", 1.0)])
    z = float(np.real(state.expectation_value(observable)))
    return float(np.clip((1.0 - z) / 2.0, 1e-7, 1 - 1e-7))


def plot_graph(spec: dict, path: Path) -> None:
    g = nx.Graph(); g.add_nodes_from(spec["nodes"]); g.add_edges_from(spec["coupling_edges"])
    pos = nx.spring_layout(g, seed=SEED)
    plt.figure(figsize=(6, 4)); nx.draw_networkx(g, pos, node_color="#9ecae1", node_size=750, edge_color="#24527a", font_weight="bold")
    plt.title(f"{spec['name']} coupling map ({spec['physical_qubits']} physical qubits)"); plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def noise_model() -> NoiseModel:
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rz", "sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.005, 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.01, 0.99]]))
    return model


def counts_probability(circuit: QuantumCircuit, simulator: AerSimulator, shots: int, seed: int) -> float:
    result = simulator.run(circuit, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    return sum(v for key, v in counts.items() if key[-1] == "1") / shots


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    scaler = MinMaxScaler(feature_range=(0, np.pi)); X_train_q = scaler.fit_transform(X_train); X_test_q = scaler.transform(X_test)
    classical = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, max_iter=1000)); t0 = time.perf_counter(); classical.fit(X_train, y_train); classical_pred = classical.predict(X_test); classical_time = time.perf_counter() - t0
    classical_acc = accuracy_score(y_test, classical_pred)

    template, x_params, theta_params = build_vqc()
    weights0 = rng.uniform(-0.2, 0.2, 2 * N_QUBITS)
    def loss(weights):
        p = np.array([probability_one(row, weights, template, x_params, theta_params) for row in X_train_q])
        return float(-np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p)))
    loss_history = []
    def tracked_loss(weights):
        value = loss(weights); loss_history.append(value); return value
    vqc_start = time.perf_counter(); initial_loss = loss(weights0)
    opt = minimize(tracked_loss, weights0, method="COBYLA", options={"maxiter": 30, "rhobeg": 0.2, "tol": 1e-3})
    final_loss = loss(opt.x); vqc_time = time.perf_counter() - vqc_start
    vqc_prob = np.array([probability_one(row, opt.x, template, x_params, theta_params) for row in X_test_q]); vqc_pred = (vqc_prob >= 0.5).astype(int); vqc_acc = accuracy_score(y_test, vqc_pred)
    if not loss_history or loss_history[0] != initial_loss: loss_history.insert(0, initial_loss)

    rows = [{"model":"LogisticRegression", "architecture":"classical", "evaluation_mode":"held_out_test", "seed":SEED, "shots":"", "accuracy":classical_acc, "loss":"", "iterations":"", "depth":"", "cx_count":"", "swap_count":"", "routing_metric":"", "runtime_s":classical_time, "notes":"Same raw features; StandardScaler fit on training data only."}, {"model":"Four-qubit VQC", "architecture":"logical", "evaluation_mode":"ideal_statevector", "seed":SEED, "shots":"", "accuracy":vqc_acc, "loss":final_loss, "iterations":len(loss_history)-1, "depth":template.depth(), "cx_count":template.count_ops().get("cx",0), "swap_count":0, "routing_metric":"logical baseline", "runtime_s":vqc_time, "notes":"Angle encoding; Ry/Rz trainable layers; Z expectation p(class=1)=(1-Z)/2."}]
    architecture_rows = []
    compiled_circuits = {}
    for filename in ["processor_A.json", "processor_B.json"]:
        spec = load_processor(filename); cmap = CouplingMap(spec["coupling_edges"])
        compiled = transpile(template, basis_gates=BASIS, coupling_map=cmap, optimization_level=1, seed_transpiler=SEED)
        ops = compiled.count_ops(); swaps = int(ops.get("swap", 0)); cx = int(ops.get("cx", 0)); layout = str(compiled.layout.initial_layout)
        routing = cx - int(template.count_ops().get("cx", 0)); compiled_circuits[spec["name"]] = (compiled, cmap)
        architecture_rows.append({"processor":spec["name"], "nodes":spec["physical_qubits"], "edges":len(spec["coupling_edges"]), "logical_qubits":N_QUBITS, "depth":compiled.depth(), "cx_count":cx, "swap_count":swaps, "routing_cx_overhead":routing, "layout":layout, "basis_gates":",".join(BASIS), "optimization_level":1, "seed_transpiler":SEED})
        rows.append({"model":"Four-qubit VQC", "architecture":spec["name"], "evaluation_mode":"compiled_ideal_metadata", "seed":SEED, "shots":"", "accuracy":"", "loss":final_loss, "iterations":len(loss_history)-1, "depth":compiled.depth(), "cx_count":cx, "swap_count":swaps, "routing_metric":routing, "runtime_s":"", "notes":"Same logical circuit; physical graph used only as coupling map."})

    for spec_name, (compiled, cmap) in compiled_circuits.items():
        sim_ideal = AerSimulator(); sim_noisy = AerSimulator(noise_model=noise_model()); noisy_preds=[]; ideal_preds=[]
        for i, row in enumerate(X_test_q):
            bound = bind_circuit(template, x_params, theta_params, row, opt.x, measure=True)
            target_ideal = transpile(bound, basis_gates=BASIS, coupling_map=cmap, optimization_level=1, seed_transpiler=SEED)
            ideal_preds.append(int(counts_probability(target_ideal, sim_ideal, SHOTS, SEED+i) >= 0.5))
            noisy_preds.append(int(counts_probability(target_ideal, sim_noisy, SHOTS, SEED+i) >= 0.5))
        ia = accuracy_score(y_test, ideal_preds); na = accuracy_score(y_test, noisy_preds)
        rows.extend([{ "model":"Four-qubit VQC", "architecture":spec_name, "evaluation_mode":"ideal_shot_based", "seed":SEED, "shots":SHOTS, "accuracy":ia, "loss":final_loss, "iterations":len(loss_history)-1, "depth":compiled.depth(), "cx_count":compiled.count_ops().get("cx",0), "swap_count":compiled.count_ops().get("swap",0), "routing_metric":"see architecture_metrics.csv", "runtime_s":"", "notes":"AerSimulator, no noise."}, {"model":"Four-qubit VQC", "architecture":spec_name, "evaluation_mode":"noisy_shot_based", "seed":SEED, "shots":SHOTS, "accuracy":na, "loss":final_loss, "iterations":len(loss_history)-1, "depth":compiled.depth(), "cx_count":compiled.count_ops().get("cx",0), "swap_count":compiled.count_ops().get("swap",0), "routing_metric":"see architecture_metrics.csv", "runtime_s":"", "notes":"Aer depolarizing errors: rz/sx/x=0.001, cx=0.005; symmetric 0.01 readout error."}])

    pd.DataFrame(rows).to_csv(RESULTS/"results.csv", index=False); pd.DataFrame(architecture_rows).to_csv(RESULTS/"architecture_metrics.csv", index=False)
    pd.DataFrame({"evaluation":range(len(loss_history)),"loss":loss_history}).to_csv(RESULTS/"training_history.csv", index=False)
    for fname in ["processor_A.json","processor_B.json"]: plot_graph(load_processor(fname), RESULTS/(Path(fname).stem.replace("processor_", "processor_")+"_coupling.png"))
    arch = pd.DataFrame(architecture_rows); plt.figure(figsize=(7,4)); plt.bar(arch.processor, arch.depth, color=["#4c78a8","#f58518"]); plt.ylabel("Compiled depth"); plt.title("Compiled architecture depth"); plt.tight_layout(); plt.savefig(RESULTS/"architecture_comparison.png", dpi=180); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(range(len(loss_history)), loss_history, marker="o", ms=2); plt.xlabel("Objective evaluation"); plt.ylabel("Binary cross-entropy"); plt.title("VQC training convergence"); plt.tight_layout(); plt.savefig(RESULTS/"training_convergence.png", dpi=180); plt.close()
    summary = pd.DataFrame([{"label":"Classical baseline","accuracy":classical_acc},{"label":"Logical VQC","accuracy":vqc_acc}] + [{"label":f"{r['processor']} ideal", "accuracy":next(x['accuracy'] for x in rows if x['architecture']==r['processor'] and x['evaluation_mode']=='ideal_shot_based')} for r in architecture_rows] + [{"label":f"{r['processor']} noisy", "accuracy":next(x['accuracy'] for x in rows if x['architecture']==r['processor'] and x['evaluation_mode']=='noisy_shot_based')} for r in architecture_rows]); plt.figure(figsize=(9,4)); plt.bar(summary.label, summary.accuracy, color="#59a14f"); plt.ylim(0,1); plt.xticks(rotation=25, ha="right"); plt.ylabel("Held-out accuracy"); plt.title("Accuracy comparison"); plt.tight_layout(); plt.savefig(RESULTS/"accuracy_comparison.png", dpi=180); plt.close()
    report = {"python":platform.python_version(),"qiskit":"1.4.2 target","qiskit_aer":"0.16.1 target","dataset":{"samples":len(X),"features":X.shape[1],"train":len(X_train),"test":len(X_test),"class_counts":np.bincount(y).tolist()},"vqc":{"logical_qubits":N_QUBITS,"initial_loss":initial_loss,"final_loss":final_loss,"optimizer_success":bool(opt.success),"optimizer_message":str(opt.message),"evaluations":len(loss_history)-1},"runtime_s":time.perf_counter()-started,"notes":"Synthetic benchmark; no clinical interpretation."}
    (RESULTS/"run_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
