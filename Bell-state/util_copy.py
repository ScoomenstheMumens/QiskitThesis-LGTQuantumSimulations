import copy
import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister,transpile,Aer,execute
from qiskit.utils.mitigation.fitters import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit_experiments.library import StateTomography
from qiskit.transpiler.passes import RemoveBarriers
def qiskit_calibration_circuits(N_qubits,qubits_measure):
    '''
    returns the measurement mitigation circuits.
    Args:
    ----
        N_qubits (int): N_qubits
        qubits_measure (list) : qubits to measure
    
    Returns:
    ----
        qc (list(QuantumCircuit)) : list of mitigation circuits
        state_labels (list) : list of state labels to pass to compleatemeasfitter
    '''
    calib_circuits = []
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

def GEM_calibration_circuits(qc,qubits_measure):
    '''
    returns the calibration circuits for the GEM and AmpDep mitigation tecnique GEM.
    Args:
    ----
        qc (QuantumCircuit) : quantum circuit to divide
        qubits_measure (list) : qubits to measure
    
    Returns:
    ----
        qc (list(QuantumCircuit)) : list of mitigation circuits
        state_labels (list) : list of state labels to pass to compleatemeasfitter
    '''

    N_qubits_measure=len(qubits_measure)
    
    calib_circuits = [[],[]]
    N_qubits = len(qc.qubits)
    state_labels = bin_list(N_qubits_measure)
    
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
        for state in state_labels:
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
    return calib_circuits, state_labels

def GEM_half_circuits(qc):
    '''
    splits the quantum circuit:
    if the number of c_nots is even than it split equally, else the first pars has 
    1 c-not less than the second.
    Args:
    ----
        qc (QuantumCircuit) : quantum circuit to divide
    
    Returns:
    ----
        counts_vector (np.array): the vector of counts.
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
    qc_half_2 = QuantumCircuit.from_qasm_str(half_2_qasm)

    return qc_half_1, qc_half_2

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

def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length number_of_qubits."""
    
    return f"{num:b}".zfill(number_of_qubits)

def bin_list(N_qubit):
    """
    Args:
    ----
        N_qubit : Number of qubits
    
    Returns:
    ----
        bin (list): binary list [0,2**(N_qubits)]
    """
    bin=[]
    for i in range(2**N_qubit):
        bin.append(DecimalToBinary(i,N_qubit))
    
    return bin

def parse_pauli_sum_op(pauli_sum_op):
    """
    Args:
    ----
        pauli_sum_op : A PauliSumOp that rapresent a pauli_string or an Hamiltonian
    
    Returns:
    ----
        pauli_strings (str): list of pauli strings operators in pauli_sum_op as str without the identities()
        pauli_qubits (int): list of qubits relative to the strings
        pauli_coeffs: list of coefficients in the Hamiltonian
    """
    pauli_strings = []
    pauli_qubits = []
    pauli_coeffs = []
    pauli_ops=pauli_sum_op.primitive.paulis
    pauli_coeffs=pauli_sum_op.primitive.coeffs
    for term in pauli_ops:
        pauli_str = term.to_label()
        # Rimuovi le I dagli elementi del PauliOp
        pauli_str_noI = ''.join([pauli if pauli != 'I' else '' for pauli in pauli_str])
        pauli_strings.append(pauli_str_noI)
        # Ottieni gli indici dei qubit in cui viene applicato il PauliOp
        pauli_qubit_idx = [idx for idx, pauli in enumerate(pauli_str) if pauli != 'I']
        pauli_qubits.append(pauli_qubit_idx)
    return pauli_strings, pauli_qubits, pauli_coeffs

def measure_pauli_string(circuit, pauli_string, target_qubits):
    """
    Args:
    ----
        pauli_string (str): Pauli string without identities
    
    Returns:
    ----
        qc (QuantumCircuit): quantum circuit for the measurement of the string 

    """
    n_qubits = circuit.num_qubits
    qr=QuantumRegister(n_qubits)
    cr=ClassicalRegister(len(target_qubits))
    qc=QuantumCircuit(qr,cr)
    qc.append(circuit,qr)
    pauli_list=[]
    pauli_list[:0] = pauli_string
    for i,pauli_str in enumerate(pauli_list):
        target_qubit = target_qubits[i]
        if pauli_str == 'X':
            qc.h(target_qubit)
        elif pauli_str == 'Y':
            qc.sdg(target_qubit)
            qc.h(target_qubit)
        elif pauli_str == 'Z':
            pass # no operation needed
        else:
            raise ValueError("Invalid Pauli string")
        qc.measure(qr[target_qubit], cr[i])
    return qc

def measure_pauli_operators_miti(circuit, pauli_op, qubit_indices):
    """
    Misura gli operatori di Pauli specificati sui qubit specificati in un circuito dato.
    
    :param circuit: il circuito quantistico a cui applicare le misure degli operatori di Pauli
    :param pauli_operators: una lista di stringhe di operatori di Pauli ('X', 'Y', o 'Z')
    :param qubit_indices: una lista di indici dei qubit sui quali applicare gli operatori di Pauli
    
    :returns: una lista di circuiti quantistici che eseguono le misure degli operatori di Pauli sui qubit specificati
    """
    # Creazione di una lista vuota per i circuiti di misura
    measurement_circuits = []
    
    n_qubits = circuit.num_qubits

    # Iterazione sugli operatori di Pauli
    measurement_circuits.append(measure_pauli_string(circuit,pauli_op,qubit_indices))
    circs_qiskit,state_labels=qiskit_calibration_circuits(n_qubits,qubits_measure=qubit_indices)
    circs_GEM,state_labels=GEM_calibration_circuits(circuit,qubits_measure=qubit_indices)
    measurement_circuits+=circs_qiskit
    measurement_circuits+=circs_GEM[0]
    measurement_circuits+=circs_GEM[1]
    for circ in measurement_circuits:
        print(transpile(circ,basis_gates=['u1', 'u2', 'u3', 'cx'],optimization_level=0).count_ops())
    
    return measurement_circuits
def measure_pauli_operators_meas(circuit, pauli_op, qubit_indices):
    """
    Misura gli operatori di Pauli specificati sui qubit specificati in un circuito dato.
    
    :param circuit: il circuito quantistico a cui applicare le misure degli operatori di Pauli
    :param pauli_operators: una lista di stringhe di operatori di Pauli ('X', 'Y', o 'Z')
    :param qubit_indices: una lista di indici dei qubit sui quali applicare gli operatori di Pauli
    
    :returns: una lista di circuiti quantistici che eseguono le misure degli operatori di Pauli sui qubit specificati
    """
    # Creazione di una lista vuota per i circuiti di misura
    measurement_circuits = []
    
    n_qubits = circuit.num_qubits

    # Iterazione sugli operatori di Pauli
    measurement_circuits.append(measure_pauli_string(circuit,pauli_op,qubit_indices))
    circs_qiskit,state_labels=qiskit_calibration_circuits(n_qubits,qubits_measure=qubit_indices)
    measurement_circuits+=circs_qiskit
      
    
    return measurement_circuits
def bootstrap_counts(counts, k, L, return_mean=False, complete_meas_fitter=None):
    """
    Prende in input un dizionario di counts e restituisce k ricampionamenti di lunghezza L.
    Se return_mean è True, restituisce invece il dizionario di counts medio dei k ricampionamenti.
    """
    shots = sum(counts.values())
    measurements = np.random.multinomial(L, [c/shots for c in counts.values()], k)
    
    if complete_meas_fitter is not None:
      if return_mean is True:
          return complete_meas_fitter.filter.apply(dict(zip(counts.keys(), np.mean(measurements, axis=0))))
      else:
          dic=[]
          for m in measurements:
            dic.append(complete_meas_fitter.filter.apply(dict(zip(counts.keys(), m))))
          return dic
    else:
      if return_mean is True:
          return dict(zip(counts.keys(), np.mean(measurements, axis=0)))
      else:
          return [dict(zip(counts.keys(), m)) for m in measurements]

def bootstrap_mitigated_expectation(counts, observable, k, L, complete_meas_fitter=None):
    """
    Prende in input un dizionario di counts, un'osservabile, il numero di ricampionamenti k,
    la lunghezza dei campioni L e un oggetto CompleteMeasFitter (opzionale).
    Restituisce la media e la deviazione bootstrap dell'osservabile su k campioni di lunghezza L,
    ognuno dei quali può essere sottoposto a mitigazione degli errori mediante il fitter fornito.
    """
    if complete_meas_fitter is not None:
        bootstrapped_counts = bootstrap_counts(counts, k, L)
        expectation_values = []
        for b_counts in bootstrapped_counts:
            expval = sum(d * p for d, p in zip(np.diag(observable), b_counts.values()))/(sum(b_counts.values()))
            expectation_values.append(expval)
    else:
        bootstrapped_counts = bootstrap_counts(counts, k, L)
        expectation_values = []
        for b_counts in bootstrapped_counts:
            expval = sum(d * p for d, p in zip(np.diag(observable), b_counts.values()))/(sum(b_counts.values()))
            expectation_values.append(expval)
    mean = np.mean(expectation_values)
    std = np.std(expectation_values, ddof=1)
    return mean, std

def estimator_meas(circuit,operator,estimator,shots=10000,level_miti=1,k=50,meas_fitters=[]):
    L=int(shots/10)
    exp_vals=[]
    exp_errors=[]
    operators,qubits,coefficients=parse_pauli_sum_op(operator)
    meas_circs=[]

    for i,operator in enumerate(operators):
        meas_circs.append(measure_pauli_string(circuit,operator,qubits[i]))
    job = estimator.run(circuits=meas_circs,parameter_values=None, parameters=None,shots=shots)
    job_result=job.result().quasi_dists
    bin_dict=[]
    for i in range (0,len(job_result)):
        b=list(job_result[i].keys())
        for j in range (0,len(b)):
            b[j]=DecimalToBinary(b[j],meas_circs[i].num_clbits)
        a=list(job_result[i].values())
        bin_dict.append(dict(zip(b,a)))
    for i,operator in enumerate(operators):
        diag_pauli_op = np.diag([(-1)**bin(k).count('1') for k in range(2**len(qubits[i]))])
        meas_calibs, s_labels = complete_meas_cal(np.arange(len(qubits[i])), len(qubits[i]), len(qubits[i]), circlabel='')
        job_cal_aux = execute(meas_calibs, backend=Aer.get_backend('aer_simulator'), shots=shots)
        cal_aux = job_cal_aux.result() 
        meas_fitter_aux = CompleteMeasFitter(cal_aux, state_labels=s_labels)
        if level_miti==1:
            meas_fitter_meas=copy.deepcopy(meas_fitter_aux)
            meas_fitter_meas.cal_matrices=meas_fitters[i][0]
            meas_fitter_GEM=copy.deepcopy(meas_fitter_aux)
            meas_fitter_GEM.cal_matrices=meas_fitters[i][1]
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_meas)
            exp_val=bootstrap_mitigated_expectation(miti_counts,diag_pauli_op,k,L,complete_meas_fitter=meas_fitter_GEM)
            exp_vals.append(exp_val[0])
            exp_errors.append(exp_val[1])
        if  level_miti==2:
            meas_fitter_meas=copy.deepcopy(meas_fitter_aux)
            meas_fitter_meas.cal_matrix=meas_fitters[i][0]
            r=meas_fitters[i][1]
            p_t=meas_fitters[i][2]
            random_vector=dict(zip(s_labels,r*shots))
            c=random_vector
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_meas)
            for s,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            exp_val_rand=bootstrap_mitigated_expectation(c,diag_pauli_op,k,L,complete_meas_fitter=None)
            exp_vals.append(exp_val_rand[0])
            exp_errors.append(exp_val_rand[1])
        if  level_miti==3:
            
            meas_fitter_meas=copy.deepcopy(meas_fitter_aux)
            meas_fitter_meas.cal_matrix=meas_fitters[i][0]
            meas_fitter_ampdep=copy.deepcopy(meas_fitter_aux)
            meas_fitter_ampdep.cal_matrix=meas_fitters[i][2]
            r=meas_fitters[i][1]
            p=meas_fitters[i][2]
            
            random_v=dict(zip(s_labels,r*shots))
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_meas)
            c=random_v
            for i,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p[i]*random_v[label])/(1-p[i])
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            exp_val=bootstrap_mitigated_expectation(c,diag_pauli_op,k,L,complete_meas_fitter=meas_fitter_ampdep)
            exp_vals.append(exp_val[0])
            exp_errors.append(exp_val[1])
    exp_val=0
    exp_err=0
    for l,coeff in enumerate(coefficients):
        exp_val+=coeff*exp_vals[l]
        exp_err+=pow(coeff,2)*pow(exp_errors[l],2)
    exp_err=np.sqrt(exp_err)

    return exp_val,exp_err

def miti_estimator(circuit,operator,estimator,shots=10000,level_miti=1,k=50):
    L=shots/10
    exp_vals=[]
    exp_errors=[]
    operators,qubits,coefficients=parse_pauli_sum_op(operator)
    meas_circs=[]
    emme=[]

    for i,operator in enumerate(operators):
        meas_circs+=measure_pauli_operators_miti(circuit,operator,qubits[i])
    job = estimator.run(circuits=meas_circs,parameter_values=None, parameters=None,shots=shots)
    job_result=job.result().quasi_dists
    bin_dict=[]
    start=0
    for i in range (0,len(job_result)):
        b=list(job_result[i].keys())
        for j in range (0,len(b)):
            b[j]=DecimalToBinary(b[j],meas_circs[i].num_clbits)
        a=list(job_result[i].values())
        bin_dict.append(dict(zip(b,a)))
    for i, pauli_op in enumerate(operators):
        diag_pauli_op = np.diag([(-1)**bin(k).count('1') for k in range(2**len(qubits[i]))])
        m_aux=[]
        meas_calibs, s_labels = complete_meas_cal(np.arange(len(qubits[i])), len(qubits[i]), len(qubits[i]), circlabel='')
        job_cal_aux = execute(meas_calibs, backend=Aer.get_backend('aer_simulator'), shots=shots)
        cal_aux = job_cal_aux.result() 
        job_qiskit=copy.deepcopy(cal_aux)
        job_GEM_L=copy.deepcopy(cal_aux)
        job_GEM_R=copy.deepcopy(cal_aux)


        result_measure=bin_dict[start]
        for j in range (0,len(s_labels)):
            job_qiskit.results[j].data.counts=bin_dict[1+start+j]
        meas_fitter_qiskit = CompleteMeasFitter(job_qiskit, state_labels=s_labels)
        Cal_qiskit=meas_fitter_qiskit.cal_matrix
        m_aux.append(Cal_qiskit)
        for j in range (0,len(s_labels)):
            job_GEM_L.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[1+start+len(s_labels)+j],method="least_squares")
            job_GEM_R.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[1+start+2*len(s_labels)+j],method="least_squares")
        start+=3*len(s_labels)+1 
        meas_fitter_GEM_L = CompleteMeasFitter(job_GEM_L, state_labels=s_labels)
        meas_fitter_GEM_R = CompleteMeasFitter(job_GEM_R, state_labels=s_labels)
        Cal_GEM_L = meas_fitter_GEM_L.cal_matrix
        Cal_GEM_R = meas_fitter_GEM_R.cal_matrix
        Cal_GEM=(Cal_GEM_L+Cal_GEM_R)/2
        if level_miti==1:
            m_aux.append(Cal_GEM)
            meas_fitter_GEM=copy.deepcopy(meas_fitter_qiskit)
            meas_fitter_GEM.cal_matrix=Cal_GEM

            miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            exp_val=bootstrap_mitigated_expectation(miti_counts,diag_pauli_op,k,L,complete_meas_fitter=meas_fitter_GEM)
            exp_vals.append(exp_val[0])
            exp_errors.append(exp_val[1])
        
        if level_miti==2:
            r=np.sum(Cal_GEM,axis=1,dtype='float')
            r=r/np.sum(r)
            m_aux.append(r)
            p_t=[]
            for q in range (0,len(r)):
                p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
            m_aux.append(p_t)
            miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            random_vector=dict(zip(s_labels,r*shots))
            c=random_vector
            for s,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            exp_val_rand=bootstrap_mitigated_expectation(c,diag_pauli_op,k,L,complete_meas_fitter=None)
            exp_vals.append(exp_val_rand[0])
            exp_errors.append(exp_val_rand[1])
        if level_miti==3:
            r=np.sum(Cal_GEM,axis=1,dtype='float')
            r=r/np.sum(r)
            m_aux.append(r)
            p_t=[]
            for q in range (0,len(r)):
                p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
            m_aux.append(p_t)
            '''
            p_t=0
            for m in range (0,len(r)):
                p_t+=(Cal_GEM[m][m]-1)/(r[m]-1)/len(r)
            #'''

            random_vector=dict(zip(s_labels,r*shots))
            Cal_ampdep=Cal_GEM
            for x in range (0,len(s_labels)):
                for y in range (0,len(s_labels)):
                    Cal_ampdep[x][y]=(Cal_GEM[x][y]-p_t[y]*r[y])/(1-p_t[y])
            m_aux.append(Cal_ampdep)
            meas_fitter_ampdep=copy.deepcopy(meas_fitter_qiskit)
            meas_fitter_ampdep.cal_matrix=Cal_ampdep
            miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            c=random_vector
            for s,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            exp_val=bootstrap_mitigated_expectation(c,diag_pauli_op,k,L,complete_meas_fitter=meas_fitter_ampdep)
            exp_vals.append(exp_val[0])
            exp_errors.append(exp_val[1])
        emme.append(m_aux)    
    exp_val=0
    exp_err=0
    for l,coeff in enumerate(coefficients):
        exp_val+=coeff*exp_vals[l]
        exp_err+=pow(coeff,2)*pow(exp_errors[l],2)
    exp_err=np.sqrt(exp_err)
    print(emme)
    print('diocane')
    return exp_val,exp_err,emme

def miti_counts(circuit,estimator,qubit_indices,shots=10000,level_miti=1,k=50):
    L=shots/10
    meas_circs=[]
    n_qubits=circuit.num_qubits
    circs_qiskit,s_labels=qiskit_calibration_circuits(n_qubits,qubits_measure=qubit_indices)
    circs_GEM,s_labels=GEM_calibration_circuits(circuit,qubits_measure=qubit_indices)
    qr=QuantumRegister(n_qubits)
    cr=ClassicalRegister(len(qubit_indices))
    qc=QuantumCircuit(qr,cr)
    qc.append(circuit,qr)
    for i,qubit in enumerate(qubit_indices):
        qc.measure(qr[qubit],cr[i])
    meas_circs.append(qc)
    meas_circs+=circs_qiskit
    meas_circs+=circs_GEM[0]
    meas_circs+=circs_GEM[1]
    job = estimator.run(circuits=meas_circs,parameter_values=None, parameters=None,shots=shots)
    job_result=job.result().quasi_dists
    bin_dict=[]
    for i in range (0,len(job_result)):
        b=list(job_result[i].keys())
        for j in range (0,len(b)):
            b[j]=DecimalToBinary(b[j],meas_circs[i].num_clbits)
        a=list(job_result[i].values())
        bin_dict.append(dict(zip(b,a)))

    m_aux=[]
    meas_calibs, s_labels = complete_meas_cal(np.arange(len(qubit_indices)), len(qubit_indices), len(qubit_indices), circlabel='')
    job_cal_aux = execute(meas_calibs, backend=Aer.get_backend('aer_simulator'), shots=shots)
    cal_aux = job_cal_aux.result() 
    job_qiskit=copy.deepcopy(cal_aux)
    job_GEM_L=copy.deepcopy(cal_aux)
    job_GEM_R=copy.deepcopy(cal_aux)


    result_measure=bin_dict[0]
    for j in range (0,len(s_labels)):
        job_qiskit.results[j].data.counts=bin_dict[1+j]
    meas_fitter_qiskit = CompleteMeasFitter(job_qiskit, state_labels=s_labels)
    Cal_qiskit=meas_fitter_qiskit.cal_matrix
    m_aux.append(Cal_qiskit)
    for j in range (0,len(s_labels)):
        job_GEM_L.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[1+len(s_labels)+j],method="least_squares")
        job_GEM_R.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[1+2*len(s_labels)+j],method="least_squares")
    meas_fitter_GEM_L = CompleteMeasFitter(job_GEM_L, state_labels=s_labels)
    meas_fitter_GEM_R = CompleteMeasFitter(job_GEM_R, state_labels=s_labels)
    Cal_GEM_L = meas_fitter_GEM_L.cal_matrix
    Cal_GEM_R = meas_fitter_GEM_R.cal_matrix
    Cal_GEM=(Cal_GEM_L+Cal_GEM_R)/2
    if level_miti==0:
        miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=None)
    if level_miti==1:
        m_aux.append(Cal_GEM)
        meas_fitter_GEM=copy.deepcopy(meas_fitter_qiskit)
        meas_fitter_GEM.cal_matrix=Cal_GEM
        miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
        miti_counts=bootstrap_counts(miti_counts,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_GEM)
    if level_miti==2:
        r=np.sum(Cal_GEM,axis=1,dtype='float')
        r=r/np.sum(r)
        m_aux.append(r)
        p_t=[]
        for q in range (0,len(r)):
            p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
        m_aux.append(p_t)
        miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
        random_vector=dict(zip(s_labels,r*shots))
        c=random_vector
        for s,label in enumerate(s_labels):
            if label not in miti_counts.keys():
                miti_counts[label]=0
            c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
        if any(v < 0 for v in c.values()):
            sum_c=sum(c.values())
            m = min(c.values())
            c = {k: v - 2 * m for k, v in c.items()}
            sum_counts=sum(c.values())
            for u in c.values():
                u=u/sum_counts*sum_c
        miti_counts=bootstrap_counts(c,20,shots,return_mean=True,complete_meas_fitter=None)
    if level_miti==3:
        r=np.sum(Cal_GEM,axis=1,dtype='float')
        r=r/np.sum(r)
        m_aux.append(r)
        p_t=[]
        for q in range (0,len(r)):
            p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
        m_aux.append(p_t)
        '''
        p_t=0
        for m in range (0,len(r)):
            p_t+=(Cal_GEM[m][m]-1)/(r[m]-1)/len(r)
        #'''

        random_vector=dict(zip(s_labels,r*shots))
        Cal_ampdep=Cal_GEM
        for x in range (0,len(s_labels)):
            for y in range (0,len(s_labels)):
                Cal_ampdep[x][y]=(Cal_GEM[x][y]-p_t[y]*r[y])/(1-p_t[y])
        m_aux.append(Cal_ampdep)
        meas_fitter_ampdep=copy.deepcopy(meas_fitter_qiskit)
        meas_fitter_ampdep.cal_matrix=Cal_ampdep
        miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
        c=random_vector
        for s,label in enumerate(s_labels):
            if label not in miti_counts.keys():
                miti_counts[label]=0
            c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
        if any(v < 0 for v in c.values()):
            sum_c=sum(c.values())
            m = min(c.values())
            c = {k: v - 2 * m for k, v in c.items()}
            sum_counts=sum(c.values())
            for u in c.values():
                u=u/sum_counts*sum_c
        miti_counts=bootstrap_counts(c,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_ampdep)
    if level_miti==4:
        miti_counts=bootstrap_counts(result_measure,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
    return miti_counts,m_aux

def miti_fid(circuit,estimator,qubit_indices,shots=10000,level_miti=1,k=50):
    L=shots/10
    meas_circs=[]
    miti_fid_counts=[]
    bin_dict=[]
    m_aux=[]
    n_qubits=circuit.num_qubits
    st_circs = StateTomography(circuit, measurement_qubits=qubit_indices).circuits()
    st_circs=transpile(st_circs,optimization_level=0,basis_gates=['u1','u2','u3','cx'])
    for circ in st_circs:
        meas_circs.append(RemoveBarriers()(circ))
    circs_qiskit,s_labels=qiskit_calibration_circuits(n_qubits,qubits_measure=qubit_indices)
    circs_GEM,s_labels=GEM_calibration_circuits(circuit,qubits_measure=qubit_indices)

    meas_circs+=circs_qiskit
    meas_circs+=circs_GEM[0]
    meas_circs+=circs_GEM[1]
    print(len(meas_circs))
    #meas_circs=transpile(meas_circs,optimization_level=0,basis_gates=['u1','u2','u3','cx'])
    job = estimator.run(circuits=meas_circs,parameter_values=None, parameters=None,shots=shots)
    job_result=job.result().quasi_dists

    for i in range (0,len(job_result)):
        b=list(job_result[i].keys())
        for j in range (0,len(b)):
            b[j]=DecimalToBinary(b[j],meas_circs[i].num_clbits)
        a=list(job_result[i].values())
        bin_dict.append(dict(zip(b,a)))

    
    meas_calibs, s_labels = complete_meas_cal(np.arange(len(qubit_indices)), len(qubit_indices), len(qubit_indices), circlabel='')
    job_cal_aux = execute(meas_calibs, backend=Aer.get_backend('aer_simulator'), shots=shots)
    cal_aux = job_cal_aux.result() 
    job_qiskit=copy.deepcopy(cal_aux)
    job_GEM_L=copy.deepcopy(cal_aux)
    job_GEM_R=copy.deepcopy(cal_aux)


    start=len(st_circs)
    for j in range (0,len(s_labels)):
        job_qiskit.results[j].data.counts=bin_dict[j+start]
    meas_fitter_qiskit = CompleteMeasFitter(job_qiskit, state_labels=s_labels)
    Cal_qiskit=meas_fitter_qiskit.cal_matrix
    m_aux.append(Cal_qiskit)
    for j in range (0,len(s_labels)):
        job_GEM_L.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[len(s_labels)+j+start],method="least_squares")
        job_GEM_R.results[j].data.counts=meas_fitter_qiskit.filter.apply(bin_dict[2*len(s_labels)+j+start],method="least_squares")
    meas_fitter_GEM_L = CompleteMeasFitter(job_GEM_L, state_labels=s_labels)
    meas_fitter_GEM_R = CompleteMeasFitter(job_GEM_R, state_labels=s_labels)
    Cal_GEM_L = meas_fitter_GEM_L.cal_matrix
    Cal_GEM_R = meas_fitter_GEM_R.cal_matrix
    Cal_GEM=(Cal_GEM_L+Cal_GEM_R)/2
    if level_miti==0:
        for i in range (0, len(st_circs)):
            miti_fid_counts.append(bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=None))
    if level_miti==1:
        m_aux.append(Cal_GEM)
        meas_fitter_GEM=copy.deepcopy(meas_fitter_qiskit)
        meas_fitter_GEM.cal_matrix=Cal_GEM
        for i in range (0, len(st_circs)):
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            miti_fid_counts.append(bootstrap_counts(miti_counts,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_GEM))
    if level_miti==2:
        r=np.sum(Cal_GEM,axis=1,dtype='float')
        r=r/np.sum(r)
        m_aux.append(r)
        p_t=[]
        for q in range (0,len(r)):
            p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
        m_aux.append(p_t)
        for i in range (0, len(st_circs)):
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            random_vector=dict(zip(s_labels,r*shots))
            c=random_vector
            for s,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
                
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            miti_fid_counts.append(bootstrap_counts(c,20,shots,return_mean=True,complete_meas_fitter=None))
    if level_miti==3:
        r=np.sum(Cal_GEM,axis=1,dtype='float')
        r=r/np.sum(r)
        m_aux.append(r)
        p_t=[]
        for q in range (0,len(r)):
            p_t.append((Cal_GEM[q][q]-1)/(r[q]-1))
        m_aux.append(p_t)
        '''
        p_t=0
        for m in range (0,len(r)):
            p_t+=(Cal_GEM[m][m]-1)/(r[m]-1)/len(r)
        #'''

        random_vector=dict(zip(s_labels,r*shots))
        Cal_ampdep=Cal_GEM
        for x in range (0,len(s_labels)):
            for y in range (0,len(s_labels)):
                Cal_ampdep[x][y]=(Cal_GEM[x][y]-p_t[y]*r[y])/(1-p_t[y])
        m_aux.append(Cal_ampdep)
        meas_fitter_ampdep=copy.deepcopy(meas_fitter_qiskit)
        meas_fitter_ampdep.cal_matrix=Cal_ampdep
        for i in range (0, len(st_circs)):
            miti_counts=bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit)
            c=random_vector
            for s,label in enumerate(s_labels):
                if label not in miti_counts.keys():
                    miti_counts[label]=0
                c[label]=(miti_counts[label]-p_t[s]*random_vector[label])/(1-p_t[s])
            
            if any(v < 0 for v in c.values()):
                sum_c=sum(c.values())
                m = min(c.values())
                c = {k: v - 2 * m for k, v in c.items()}
                sum_counts=sum(c.values())
                for u in c.values():
                    u=u/sum_counts*sum_c
            booty=bootstrap_counts(c,20,shots,return_mean=True,complete_meas_fitter=meas_fitter_ampdep)
            if any(v < 0 for v in booty.values()):
                sum_c=sum(booty.values())
                m = min(booty.values())
                booty = {k: v - 2 * m for k, v in booty.items()}
                sum_counts=sum(booty.values())
                for u in booty.values():
                    u=u/sum_counts*sum_c
            miti_fid_counts.append(booty)
    if level_miti==4:
        for i in range (0, len(st_circs)):
            miti_fid_counts.append(bootstrap_counts(bin_dict[i],20,shots,return_mean=True,complete_meas_fitter=meas_fitter_qiskit))
    return miti_fid_counts,m_aux,st_circs

