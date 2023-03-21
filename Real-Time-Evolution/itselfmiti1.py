import warnings

import numpy as np
import qiskit
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                    transpile)
from qiskit.circuit.random import random_circuit
from qiskit.ignis.verification.tomography import (StateTomographyFitter,
                                                  state_tomography_circuits)
from qiskit.quantum_info import Operator, state_fidelity
from qiskit.utils.mitigation.fitters import CompleteMeasFitter

#from lib import pauli_twirling
#from lib import convenience

def get_calibration_circuits(backend,qc, method="GEM",qubits_measure=[0,1,2]):
    print('building')
    '''
    Returns a list of calibration circuits for all the methods: CIC, NIC, LIC and qiskit calibration matrix.
    Args
    ----
        qc (QuantumCircuit): the quantum circuit you wont to calibrate.
        method (string): the method of calibration. Can be RIC, GEM or qiskit.
    Return
    ----
        calib_circuits (list of QuantumCircuit): list of calibration circuits.
    '''
    
    if method=="qiskit":
        return qiskit_calibration_circuits(N_qubits=len(qc.qubits),qubits_measure=qubits_measure)
    elif method=="GEM":
        return GEM_calibration_circuits(backend,qc=qc,qubits_measure=qubits_measure)
    elif method=="GEM_tens":
        return GEM_tens_cal_circs(backend,qc=qc,qubits_measure=qubits_measure)
   
def rand_target(val, N):
    lista = list(np.arange(N))
    lista.remove(val)
    return np.random.choice(lista)

def random_single_qubit_gate():
    '''
    return a random single qubit gate chosen between "rz", "x" and "sx"
    '''
    p = np.random.choice(["rz", "x", "sx"])
    qc = QuantumCircuit(1)
    if p=="rz":
        alpha = np.random.random()*2*np.pi
        qc.rz(alpha, 0)
    elif p=="x":
        qc.x(0)
    elif p=="sx":
        qc.sx(0)
    return qc

def random_single_qubit_gate1():
    '''
    return a random single qubit gate chosen between "rz", "x" 
    '''
    p = np.random.choice(["rz", "x"])
    qc = QuantumCircuit(1)
    if p=="rz":
        alpha = np.random.random()*2*np.pi
        qc.rz(alpha, 0)
    elif p=="x":
        qc.x(0)
    return qc

def circuit_random(N_qubits, N_cnots, N_single_gates, basis_gates=["cx", "x", "sx", "rz", "id"]):
    '''
    returns a random quantum circuit
    '''
    control_qubits = np.random.randint(N_qubits, size=N_cnots)
    target_qubits = np.array([rand_target(j, N_qubits)  for j in control_qubits])

    single_positions = np.random.randint(N_cnots+1, size=N_single_gates)
    single_qubits = np.random.randint(N_qubits, size=N_single_gates)
    qr = QuantumRegister(N_qubits, name="q")
    qc = QuantumCircuit(qr, name="random_circuit")
    j=0
    for i in range(N_cnots):
        for _ in range(list(single_positions).count(i)):
            qc.append(random_single_qubit_gate(), [qr[single_qubits[j]]])
            j+=1
        qc.cx(control_qubits[i], target_qubits[i])

    for _ in range(list(single_positions).count(N_cnots)):
        qc.append(random_single_qubit_gate(), [qr[single_qubits[j]]])
        j+=1
    return qc.decompose()#transpile(qc, basis_gates = basis_gates, optimization_level=0)

####### calibration circuits


def qiskit_calibration_circuits(N_qubits,qubits_measure=[0,2,4,6]):
    calib_circuits = []
    #state_labels = bin_list(N_qubits)
    N_qubits_measure=len(qubits_measure)
    state_labels1=bin_list(N_qubits_measure)
    for state in state_labels1:
        cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
        qr_cal = QuantumRegister(N_qubits, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        # prepares the state.
        for qubit in range(N_qubits_measure):
            if state[::-1][qubit] == "1":
                qc_cal.x(qr_cal[qubits_measure[qubit]])
        for j in range(N_qubits_measure):
            qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
        calib_circuits.append(qc_cal)
    return calib_circuits, state_labels1

def GEM_calibration_circuits(qc,qubits_measure=[0,1,2,3,4]):
    '''
    returns the calibration circuits for the mitigation tecnique GEM.
    '''

    N_qubits_measure=len(qubits_measure)
    
    calib_circuits = [[],[]]
    N_qubits = len(qc.qubits)
    state_labels1 = bin_list(N_qubits_measure)
    
    qc_half_1, qc_half_2 = GEM_half_circuits(qc)
    # first half
    qr_1 = QuantumRegister(N_qubits, name="q")
    qc_cal_1 = QuantumCircuit(qr_1, name="cal_1")
    qc_cal_1.append(qc_half_1, qr_1)
    qc_cal_1.append(qc_half_1.inverse(), qr_1)
    
    qr_2 = QuantumRegister(N_qubits, name="q")
    qc_cal_2 = QuantumCircuit(qr_2, name="cal_2")
    qc_cal_2.append(qc_half_2, qr_2)
    qc_cal_2.append(qc_half_2.inverse(), qr_2)
    
    half_circuits = [qc_cal_1, qc_cal_2]
    # prepare all the initial state.
    for i in range(2):
        for state in state_labels1:
            cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
            qr_cal = QuantumRegister(N_qubits, name = "q_")
            qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
            
            for qubit in range(N_qubits_measure):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubits_measure[qubit]])
            # than we append the circuit
            qc_cal.append(half_circuits[i], qr_cal)

            for j in range(N_qubits_measure):
                qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
            calib_circuits[i].append(qc_cal)
    return calib_circuits, state_labels1




def GEM_calibration_circuits_QPE(BACK,qc1,qc2,qubits_measure=[0,2,4,6]):
    q1_r=QuantumRegister(7)
    q2_r=QuantumRegister(7)
    f1_r=QuantumRegister(7)
    f2_r=QuantumRegister(7)
    q1=QuantumCircuit(q1_r)
    q2=QuantumCircuit(q2_r)
    f1=QuantumCircuit(f1_r)
    f2=QuantumCircuit(f2_r)
    q1,q2=GEM_half_circuits(BACK,qc1)
    f1,f2=GEM_half_circuits(BACK,qc2)
    N_qubits_measure=len(qubits_measure)
    N_qubits = len(qc1.qubits)
    state_labels1 = bin_list(N_qubits_measure)
    c=[[],[],[],[]]
    # first circuit
    qr_1 = QuantumRegister(N_qubits, name="q")
    qc_cal_1 = QuantumCircuit(qr_1, name="cal_1")
    qc_cal_1.append(q1, qr_1)
    qc_cal_1.append(q1.inverse(), qr_1)
    qc_cal_1.append(f1,qr_1)
    qc_cal_1.append(f1.inverse(), qr_1)

    #second circuit
    qr_2 = QuantumRegister(N_qubits, name="q")
    qc_cal_2 = QuantumCircuit(qr_2, name="cal_2")
    qc_cal_2.append(q2.inverse(), qr_2)
    qc_cal_2.append(q2, qr_2)
    qc_cal_2.append(f2.inverse(),qr_2)
    qc_cal_2.append(f2, qr_2)

    #third circuit
    qr_3 = QuantumRegister(N_qubits, name="q")
    qc_cal_3 = QuantumCircuit(qr_3, name="cal_3")
    qc_cal_3.append(q1, qr_3)
    qc_cal_3.append(q1.inverse(), qr_3)
    qc_cal_3.append(f2.inverse(),qr_3)
    qc_cal_3.append(f2, qr_3)

    #fourth circuit
    qr_4 = QuantumRegister(N_qubits, name="q")
    qc_cal_4 = QuantumCircuit(qr_4, name="cal_4")
    qc_cal_4.append(q2.inverse(), qr_4)
    qc_cal_4.append(q2, qr_4)
    qc_cal_4.append(f1,qr_4)
    qc_cal_4.append(f1.inverse(), qr_4)

    back_qc_cal_1=transpile(qc_cal_1,BACK,optimization_level=0)
    back_qc_cal_2=transpile(qc_cal_2,BACK,optimization_level=0)
    back_qc_cal_3=transpile(qc_cal_3,BACK,optimization_level=0)
    back_qc_cal_4=transpile(qc_cal_4,BACK,optimization_level=0)
    print(back_qc_cal_1.count_ops())
    print(back_qc_cal_2.count_ops())
    print(back_qc_cal_3.count_ops())
    print(back_qc_cal_4.count_ops())
    half_circuits = [back_qc_cal_1, back_qc_cal_2,back_qc_cal_3,back_qc_cal_4]
    # prepare all the initial state.
    for i in range(4):
        for state in state_labels1:
            cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
            qr_cal = QuantumRegister(N_qubits, name = "q_")
            qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
                
            for qubit in range(N_qubits_measure):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubits_measure[qubit]])
            # than we append the circuit
            qc_cal.append(half_circuits[i], qr_cal)

            for j in range(N_qubits_measure):
                qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
            c[i].append(qc_cal)
    return c, state_labels1   


def GEM_half_circuits(qc):
    '''
    this function splits the quantum circuit qc into two qauntum circuit:
    if the number of c_nots is even than it split equally, else the first pars has 
    1 c-not less than the second.
    '''
    try:
        N_cnots = qc.count_ops()["cx"]
    except:
        N_cnots = 0
    splitted_qasm = qc.qasm().split(";\n")
    splitted_qasm.remove("")
    half_1_qasm = ""
    half_2_qasm = ""
    i=0
    for j, element in enumerate(splitted_qasm):
        if "cx" in element:
            i+=1
        if j<3:
            half_1_qasm+=element+";\n"
            half_2_qasm+=element+";\n"
        else:
            if i<int(N_cnots/2+1):
                half_1_qasm+=element+";\n"
            else:
                if j!=len(splitted_qasm)-1:
                    half_2_qasm+=element+";\n"
    qc_half_1 = QuantumCircuit.from_qasm_str(half_1_qasm)
    #back_qc_half_1=transpile(qc_half_1,backend,optimization_level=0)
    qc_half_2 = QuantumCircuit.from_qasm_str(half_2_qasm)
    #back_qc_half_2=transpile(qc_half_2,backend,optimization_level=0)
    return qc_half_1, qc_half_2


####### fidelity and distance

def distance(counts, vector):
    '''
    this function computes the distance between the ideal vector and the counts (mitigated or not) obtained
    runing multiple times the circuit.
    '''
    #norm = sum(counts.values())
    counts_vector = occurrences_to_vector(counts)/sum(counts.values())
    #if sum(counts_vector)!=0 or sum(vector)!=0:
    #    print("normalizzare!!!", sum(counts_vector), sum(vector))
    #    print(np.dot(counts_vector-vector, counts_vector-vector)**(1/2))

    return np.dot(counts_vector-vector, counts_vector-vector)**(1/2)

def fidelity_count(result, qcs, target_state):
    '''
    given job result, tomography circuits and targhet state it returns the fidelity score.
    this function is a copy of 'fidelity_count' defined in the main commit. 
    It's used only in the time evolution notebook.
    '''
    tomo_ising = StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method="lstsq")
    fid=state_fidelity(rho_fit_ising, target_state)
    return fid

######## utilities

def occurrences_to_vector(occurrences_dict):
    """Converts the occurrences dict to vector.
    Args:
    ----
        occurrences_dict (dict) : dict returned by BaseJob.results.get_counts() 
    
    Returns:
    ----
        counts_vector (np.array): the vector of counts.
    """
    counts_vector = np.zeros(2**len(list(occurrences_dict.keys())[0]))
    for state in list(occurrences_dict.keys()):
        counts_vector[int(state, 2)] = occurrences_dict[state]
    return counts_vector

def explicit_circuit(quantum_circuit):
    '''
    returns the explicit circuit of a 'quantum_circuit'
    
    splitted_qasm = quantum_circuit.qasm().split(";\n")
    qr_explicit = QuantumRegister(len(quantum_circuit.qubits), name="q")
    qc_explicit = QuantumCircuit(qr_explicit, name="qc")
    for element in splitted_qasm:
        if "cx" in element:
            el = element.replace("q[", "").replace("]", "")
            control_target = el.split(" ")[1].split(",") '''
    qc_qasm = quantum_circuit.qasm()
    return QuantumCircuit.from_qasm_str(qc_qasm)


    
def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r



def GEM_tens_cal_circs(backend,qc,qubits_measure=[0,2,4,6]):
    '''
    returns the calibration circuits for the mitigation tecnique GEM.
    '''

    N_qubits_measure=len(qubits_measure)
    
    calib_circuits_0 = [[],[]]
    calib_circuits_1 = [[],[]]
    N_qubits = len(qc.qubits)
    
    qc_half_1, qc_half_2 = GEM_half_circuits(backend,qc=qc)
    # first half
    qr_1 = QuantumRegister(N_qubits, name="q")
    qc_cal_1 = QuantumCircuit(qr_1, name="cal_1")
    qc_cal_1.append(qc_half_1, qr_1)
    qc_cal_1.append(qc_half_1.inverse(), qr_1)
    
    qr_2 = QuantumRegister(N_qubits, name="q")
    qc_cal_2 = QuantumCircuit(qr_2, name="cal_2")
    qc_cal_2.append(qc_half_2, qr_2)
    qc_cal_2.append(qc_half_2.inverse(), qr_2)
    
    half_circuits = [qc_cal_1, qc_cal_2]
    # prepare all the initial state.
    for i in range(2):
        for qubit in N_qubits_measure:
            cr_cal = ClassicalRegister(1, name = "c")
            qr_cal = QuantumRegister(N_qubits, name = "q_")
            qc_cal = QuantumCircuit(qr_cal, cr_cal, name="f_mcalcal_{qubit}}")
            qc_cal.x(qr_cal[qubits_measure[qubit]])
            # than we append the circuit
            qc_cal.append(half_circuits[i], qr_cal)
            qc_cal.measure(qr_cal[qubits_measure[qubit]],cr_cal)
            calib_circuits_1[i].append(qc_cal)
    for i in range(2):
        for qubit in N_qubits_measure:
            cr_cal = ClassicalRegister(1, name = "c")
            qr_cal = QuantumRegister(N_qubits, name = "q_")
            qc_cal = QuantumCircuit(qr_cal, cr_cal, name="f_mcalcal_{qubit}}")
            # than we append the circuit
            qc_cal.append(half_circuits[i], qr_cal)
            qc_cal.measure(qr_cal[qubits_measure[qubit]],cr_cal)
            calib_circuits_0[i].append(qc_cal)
    return calib_circuits_0,calib_circuits_1

def ansatz_cal(qc,qubits,backend):
    num_param=qc.num_parameters
    randomlist = []
    #config=backend.configuration()
    #N_qubits_backend=config.n_qubits
    N_qubits=len(qubits)
    for i in range(0,num_param):
        n = random.randint(0,100)
        n=2*n*np.pi/100
        randomlist.append(n)
    print(randomlist)
    qr_cal=QuantumRegister(N_qubits)
    qc_cal=QuantumCircuit(qr_cal)
    
    qc_cal.append(qc,qr_cal)
    qc_cal_bound=qc_cal.assign_parameters(randomlist)
    #qc_cal=transpile(qc_cal,backend,optimization_level=0)
    return qc_cal_bound
