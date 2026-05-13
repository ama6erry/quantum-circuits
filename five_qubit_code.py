import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=9)

STABILISERS = [
    ([0, 3], [1, 2]),  # g1
    ([1, 4], [2, 3]),  # g2
    ([0, 2], [3, 4]),  # g3
    ([1, 3], [0, 4]),  # g4
]

SYNDROME_TABLE = {
    (0, 0, 0, 0): None, 
    (0, 0, 0, 1): (0, 'X'), 
    (1, 0, 1, 1): (0, 'Y'), 
    (1, 0, 1, 0): (0, 'Z'), 
    (1, 0, 0, 0): (1, 'X'), 
    (1, 1, 0, 1): (1, 'Y'), 
    (0, 1, 0, 1): (1, 'Z'), 
    (1, 1, 0, 0): (2, 'X'), 
    (1, 1, 1, 0): (2, 'Y'), 
    (0, 0, 1, 0): (2, 'Z'), 
    (0, 1, 1, 0): (3, 'X'), 
    (1, 1, 1, 1): (3, 'Y'), 
    (1, 0, 0, 1): (3, 'Z'), 
    (0, 0, 1, 1): (4, 'X'), 
    (0, 1, 1, 1): (4, 'Y'), 
    (0, 1, 0, 0): (4, 'Z')
}

def build_logical_states():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    g1 = kron_all([X, Z, Z, X, I])
    g2 = kron_all([I, X, Z, Z, X])
    g3 = kron_all([X, I, X, Z, Z])
    g4 = kron_all([Z, X, I, X, Z])
    X_L = kron_all([X, X, X, X, X])

    I32 = np.eye(32, dtype=complex)
    projector = (I32 + g1) @ (I32 + g2) @ (I32 + g3) @ (I32 + g4) / 16

    zero5 = np.zeros(32, dtype=complex)
    zero5[0] = 1
    zero_L = projector @ zero5
    zero_L /= np.linalg.norm(zero_L)
    one_L = X_L @ zero_L
    return zero_L, one_L


ZERO_L, ONE_L = build_logical_states()


def encode_five_qubit(logical_state):
    state = ZERO_L if logical_state == 0 else ONE_L
    qml.StatePrep(state, wires=[0, 1, 2, 3, 4])


def measure_syndrome():
    for i, (x_sup, z_sup) in enumerate(STABILISERS):
        ancilla = 5 + i
        qml.Hadamard(wires=ancilla)
        for q in x_sup:
            qml.CNOT(wires=[ancilla, q])  # controlled-X
        for q in z_sup:
            qml.CZ(wires=[ancilla, q])    # controlled-Z
        qml.Hadamard(wires=ancilla)

    return qml.measure(5), qml.measure(6), qml.measure(7), qml.measure(8)


def syndrome_matches(s0, s1, s2, s3, target):
    # Returns 1 iff the measured syndrome (s0,s1,s2,s3) equals target. Invert each bit when target=0.
    t0, t1, t2, t3 = target
    m0 = s0 if t0 else ~s0
    m1 = s1 if t1 else ~s1
    m2 = s2 if t2 else ~s2
    m3 = s3 if t3 else ~s3
    return m0 & m1 & m2 & m3


@qml.qnode(dev)
def five_qubit_circuit(logical_state, x_error, z_error):
    encode_five_qubit(logical_state)

    # Simulate error by inserting pauli gate on wire after encoding
    if x_error is not None:
        qml.PauliX(wires=x_error)
    if z_error is not None:
        qml.PauliZ(wires=z_error)

    s0, s1, s2, s3 = measure_syndrome()

    # For syndrome pattern, conditionally apply the matching correction.
    for syndrome, correction in SYNDROME_TABLE.items():
        if correction is None:
            continue
        qubit, err_type = correction
        cond = syndrome_matches(s0, s1, s2, s3, syndrome)
        if err_type == 'X':
            qml.cond(cond, qml.PauliX)(wires=qubit)
        elif err_type == 'Z':
            qml.cond(cond, qml.PauliZ)(wires=qubit)
        elif err_type == 'Y':
            qml.cond(cond, qml.PauliY)(wires=qubit)

    # Read out the logical qubit by measuring logical Z = ZZZZZ.
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))


print("5-Qubit Code Demo")

test_params = [
    (0, None, None, "No error |0⟩"),
    (1, None, None, "No error |1⟩"),
    (0,    2, None, "X error qubit 2 |0⟩"),
    (1,    4, None, "X error qubit 4 |1⟩"),
    (0, None,    1, "Z error qubit 1 |0⟩"),
    (1, None,    3, "Z error qubit 3 |1⟩"),
    (0,    0,    0, "Y error qubit 0 |0⟩"),
    (1,    3,    3, "Y error qubit 3 |1⟩"),
]

for logical_state, x_error, z_error, description in test_params:
    z_L = five_qubit_circuit(logical_state, x_error, z_error)
    # logical <Z> = +1 means logical |0>, -1 means logical |1>
    # Convert to probabilities: P0 = (1 + <Z_L>) / 2, P1 = (1 - <Z_L>) / 2
    p0 = (1 + z_L) / 2
    p1 = (1 - z_L) / 2
    outcome_state = 0 if p0 > 0.5 else 1

    print(f"Test: {description}")
    print(f"Expected state : {logical_state} - Outcome state : {outcome_state} - P0 = {p0:.4f} - P1 = {p1:.4f}")
    print("-" * 80)
