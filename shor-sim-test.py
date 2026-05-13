from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# My submission to pennylanes A Shor Thing challenge
N_QUBITS = 9
N_SHOTS = 1500
PAULIS = ['x', 'y', 'z']

def build_shor_circuit(init_state, errors):
    alpha, beta = init_state
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    qc.initialize([alpha, beta], 0)

    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)

    qc.barrier()
    for pauli, qubit in errors:
        getattr(qc, pauli)(qubit)   # qc.x(q) / qc.y(q) / qc.z(q)
    qc.barrier()

   
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)

    qc.ccx(1, 2, 0)
    qc.ccx(4, 5, 3)
    qc.ccx(7, 8, 6)

    qc.h(0)
    qc.h(3)
    qc.h(6)

    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.ccx(3, 6, 0)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    return qc

def randomise_errors(n_qubits, p):
    errors = []
    for q in range(n_qubits):
        if random.random() < p: #Checks if error occurs based on probabilty p
            errors.append((random.choice(PAULIS), q)) #If an error does occur for that qubit, add it to list of errors
        
    return errors

def ideal_qubit0_outcome(init_state):
    alpha, _ = init_state
    if abs(alpha) ** 2 >= 0.5:
        return 0
    else:
        return 1

def simulate():
    state = (1.0, 0.0)
    p_values = np.linspace(0, 0.3, num = 60)
    n_shots = N_SHOTS

    simulator = AerSimulator()
    results = defaultdict(list)
    ideal_bit = ideal_qubit0_outcome(state)

    for p in p_values:
        shor_circuits = []
        shor_failures = 0
        for _ in range(n_shots):
            errors = randomise_errors(N_QUBITS, p)       
            shor_circuits.append(build_shor_circuit(state, errors))


        shor_job = simulator.run(shor_circuits, shots=1)
        shor_results = shor_job.result()

        
        for i in range(n_shots):
            counts = shor_results.get_counts(i)
            bitstring = next(iter(counts))
            if int(bitstring[-1]) != ideal_bit:
                shor_failures += 1

        lone_failures = 0
        for i in range(n_shots):
            if random.random() < p:
                lone_failures += 1

        p_single_fail = lone_failures / n_shots
        p_shor_fail = shor_failures / n_shots 

        print(f"p={p:.3f} | lone qubit: {p_single_fail:.4f} | shor: {p_shor_fail:.4f}")

        results["p"].append(p)
        results["single_fail"].append(p_single_fail)
        results["shor_fail"].append(p_shor_fail)

    return dict(results)

def plot_results(results, save_path = "shor_error_comparison_best.png"):

    single_fit = np.polyfit(results["p"], results["single_fail"], 1)
    shor_fit = np.polyfit(results["p"], results["shor_fail"], 2)

    single_line = np.poly1d(single_fit)
    shor_line = np.poly1d(shor_fit)

   
    plt.figure(figsize=(8,5))
    plt.plot(results["p"], results["single_fail"], label="Single qubit", alpha = 0.3, color="blue", linewidth = 1.2)
    plt.plot(results["p"], results["shor_fail"], label="Shor's code", alpha = 0.3, color="red", linewidth = 1.2)

    plt.plot(
        results["p"],
        single_line(results["p"]),
        linestyle="--",
        label="Single qubit best fit",
        color="navy",
        linewidth = 1.5
    )

    plt.plot(
        results["p"],
        shor_line(results["p"]),
        linestyle="--",
        label="Shor best fit",
        color="darkred",
        linewidth = 1.5
    )


    plt.xlabel("Probabilty of an error on any single qubit (p)")
    plt.ylabel("Probabilty of Logical failure")

    plt.title("Shor Code vs Single Qubit Error Rate")

    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi = 300, bbox_inches="tight")

    plt.show()    
    
data = simulate()
plot_results(data)