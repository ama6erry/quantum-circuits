import pennylane as qml

dev = qml.device("default.qubit", wires=13)
STABILISER_SUPPORT = [ #Hamming code (if a qubit is 1 in the row, the array has its index stored in that row)
    [0, 2, 4, 6],   # row 0
    [1, 2, 5, 6],   # row 1
    [3, 4, 5, 6],   # row 2
]

def encode_steane(logical_state):
    if logical_state == 1:
        qml.PauliX(wires=0)

    for ancilla_q in [1, 2, 3]:
        qml.Hadamard(wires=ancilla_q)

    qml.CNOT(wires=[1, 4])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[2, 5])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[3, 5])
    qml.CNOT(wires=[3, 6])
    qml.CNOT(wires=[1, 6])
    qml.CNOT(wires=[2, 6])

def measure_syndrome():
    for qubit in [10, 11, 12]: #Hadamard X stabilizers
        qml.Hadamard(wires=qubit)

    for i, stabilizer in enumerate(STABILISER_SUPPORT): #Z stabilizers
        ancilla_index = 7 + i
        for qubit in stabilizer:
            qml.CNOT(wires = [ancilla_index, qubit])
    
    for i, stabilizer in enumerate(STABILISER_SUPPORT):
        ancilla_index = 10 + i
        for qubit in stabilizer:
            qml.CNOT(wires = [qubit, ancilla_index])
    
    for qubit in [10, 11, 12]:
        qml.Hadamard(wires = qubit) #Reverse the earlier Hadamard operation

    return (qml.measure(7),  qml.measure(8),  qml.measure(9), qml.measure(10), qml.measure(11), qml.measure(12))  #Error syndromes

def syndrome_matches(s0, s1, s2, qubit):
    ###
    #Check whether the measured syndrome corresponds to a given qubit. (Look up table method)

    #Syndrome mapping:
        #qubit 0 -> 001
        #qubit 1 -> 010
        #qubit 2 -> 011
        #qubit 3 -> 100
        #qubit 4 -> 101
        #qubit 5 -> 110
        #qubit 6 -> 111
    

    # Convert qubit index into its expected 3-bit syndrome
    syndrome = qubit + 1

    expected_s0 = syndrome & 1
    expected_s1 = (syndrome >> 1) & 1
    expected_s2 = (syndrome >> 2) & 1

    match_s0 = s0 if expected_s0 else ~s0
    match_s1 = s1 if expected_s1 else ~s1
    match_s2 = s2 if expected_s2 else ~s2

    # All syndrome bits must match
    return match_s0 & match_s1 & match_s2

@qml.qnode(dev)
def steane_circuit(logical_state, x_error, z_error):
    #Encode qubit into logical qubit made up of 7 qubits
    encode_steane(logical_state)

    #Simulate errors
    if x_error is not None:
        qml.PauliX(wires = x_error)
    if z_error is not None:
        qml.PauliZ(wires = z_error)

    s_x0, s_x1, s_x2, s_z0, s_z1, s_z2 = measure_syndrome() #Error may have occurred so we measure the syndromes now

    for qubit in range(7): #After measuring the syndromes, if the syndrome indicates that an error occurred, apply the correct fix to that qubit
        qml.cond(syndrome_matches(s_x0, s_x1, s_x2, qubit), qml.PauliX)(wires = qubit)
        qml.cond(syndrome_matches(s_z0, s_z1, s_z2, qubit), qml.PauliZ)(wires = qubit)

    #Decode (just the reverse of the encode in encode_steane)
    qml.CNOT(wires=[2, 6])
    qml.CNOT(wires=[1, 6])
    qml.CNOT(wires=[3, 6])
    qml.CNOT(wires=[3, 5])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[2, 5])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[1, 4])

    for qubit in [1, 2, 3]:
        qml.Hadamard(wires = qubit)

    return qml.probs(wires = 0)

print("Steane Code Demo")

test_params = [
        # (logical_state, x_error, z_error, description)
        (0, None, None, "No error |0⟩"),
        (1, None, None, "No error |1⟩"),
        (0,    3, None, "X error qubit 3 |0⟩"),
        (1,    5, None, "X error qubit 5 |1⟩"),
        (0, None,    2, "Z error qubit 2 |0⟩"),
        (1, None,    6, "Z error qubit 6 |1⟩"),
        (0,    1,    1, "Y error qubit 1 |0⟩"),
    ]

for logical_state, x_error, z_error, description in test_params:
    probs = steane_circuit(logical_state, x_error, z_error)
    p0 = probs[0]
    p1 = probs[1]

    outcome_state = 1

    if p0 > 0.5:
        outcome_state = 0

    print(f"Test: {description}")
    print(f"Expected state : {str(logical_state)} - Outcome state : {str(outcome_state)} - P0 = {p0} = P1 = {p1}")
    print("-"*80)

    