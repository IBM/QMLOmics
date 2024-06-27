import numpy as np
import pandas as pd
import time 
import matplotlib
from glob import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('dark')

# ====== Scikit-learn imports ======

from sklearn.svm import SVC
from sklearn.metrics import (
    auc,
    roc_curve,
    ConfusionMatrixDisplay,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

# ====== Qiskit imports ======

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def compute_svc(X_train, y_train, X_test, y_test, c = 1):
    beg_time = time.time()
    svc = SVC(C=c)
    # y_train = torch.argmax(torch.tensor(y_train, dtype=torch.float32),dim=1)
    # y_test = torch.argmax(torch.tensor(y_test, dtype=torch.float32),dim=1)
    svc_vanilla = svc.fit(X_train, y_train)
    labels_vanilla = svc_vanilla.predict(X_test)
    f1_svc = f1_score(y_test, labels_vanilla, average='micro')
    print("Time taken for SVC (secs): ", time.time() - beg_time)
    print("F1 SVC: ", f1_svc)
    return f1_svc

def compute_QSVC(X_train, y_train, X_test, y_test, encoding='ZZ', c = 1, pegasos=True):
    beg_time = time.time()
    #service = QiskitRuntimeService(instance="accelerated-disc/internal/default")    
    service = QiskitRuntimeService()    
    backend = AerSimulator(method='statevector')
    algorithm_globals.random_seed = 12345

    feature_map = None

    if encoding == 'ZZ' :
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2, 
                            entanglement='linear')
    else: 
        if encoding == 'Z': 
            feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2)
        if encoding == 'P': 
            feature_map = PauliFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2, entanglement='linear')

    sampler = Sampler() 
    fidelity = ComputeUncompute(sampler=sampler)
    Qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=Qkernel, C=c)
    qsvc_model = qsvc.fit(X_train, y_train)
    labels_qsvc = qsvc_model.predict(X_test)
    f1_qsvc = f1_score(y_test, labels_qsvc, average='micro')
    
    f1_peg_qsvc = 0
    if pegasos == True: 
        peg_qsvc = PegasosQSVC(quantum_kernel=Qkernel, C=c)
        peg_qsvc_model = peg_qsvc.fit(X_train, y_train)
        labels_peg_qsvc = peg_qsvc_model.predict(X_test)
        f1_peg_qsvc = f1_score(y_test, labels_peg_qsvc, average='micro')
        
    print("Time taken for QSVC (secs): ", time.time() - beg_time)
    print("F1 QSVC: ", f1_qsvc)
    
    return f1_qsvc,f1_peg_qsvc

def compute_estimator_QNN(X_train, y_train, X_test, y_test, primitive: str):
    beg_time = time.time()
    if primitive == 'estimator':
        # construct QNN with the QNNCircuit's default ZZFeatureMap feature map and RealAmplitudes ansatz.
        qc_qnn = QNNCircuit(num_qubits=X_train.shape[1])

        estimator_qnn = EstimatorQNN(circuit=qc_qnn)
        # QNN maps inputs to [-1, +1]
        estimator_qnn.forward(X_train[0, :], algorithm_globals.random.random(estimator_qnn.num_weights))
        # construct neural network classifier
        estimator_classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=100))
        # fit classifier to data
        estimator_classifier.fit(X_train, y_train)
        f1_score_estimator_qnn = estimator_classifier.score(X_test, y_test)
        
        print("Time taken for Sampler QNN (secs): ", time.time() - beg_time)
        print("F1 Estimator QNN: ", f1_score_estimator_qnn)
        
        return f1_score_estimator_qnn
    
    if primitive == 'sampler':
        # construct a quantum circuit from the default ZZFeatureMap feature map and a customized RealAmplitudes ansatz
        qc_sampler = QNNCircuit(ansatz=RealAmplitudes(X_train.shape[1], reps=1))
        # parity maps bitstrings to 0 or 1
        def parity(x):
            return "{:b}".format(x).count("1") % 2
        output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping
        # construct QNN
        sampler_qnn = SamplerQNN(circuit=qc_sampler, interpret=parity,output_shape=output_shape,)
        # construct classifier
        sampler_classifier = NeuralNetworkClassifier(neural_network=sampler_qnn, optimizer=COBYLA(maxiter=100))
        # fit classifier to data
        sampler_classifier.fit(X_train, y_train)
        f1_score_sampler_qnn = sampler_classifier.score(X_test, y_test)
        
        print("Time taken for Sampler QNN (secs): ", time.time() - beg_time)
        print("F1 Sampler QNN: ", f1_score_sampler_qnn)
        
        return f1_score_sampler_qnn
    

if __name__ == "__main__":
    path = '/dccstor/boseukb/CuNA/data/BrCa/'
    all_files = []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            all_files.append(f)
            
    results_dict = {}

for i in range(1,11): 
    print('Iteration ', i)
    fs = [x for x in all_files if 'iter'+str(i)+'_' in x]
    f_train = [x for x in fs if 'train' in x][0]
    f_test = [x for x in fs if 'test' in x][0]
    df_train = pd.read_csv(path+f_train)
    df_test = pd.read_csv(path+f_test)
    
    X_train = df_train[[str(x) for x in list(range(10))]].values
    y_train = df_train['y'].map({'LumA':0, 'LumB': 1}).values
    X_test = df_test[[str(x) for x in list(range(10))]].values
    y_test = df_test['y'].map({'LumA':0, 'LumB': 1}).values 
    c = 10 
    
    f1_svc = compute_svc(X_train, y_train, X_test, y_test, c)
    f1_qsvc, f1_peg_qsvc = compute_QSVC(X_train, y_train, X_test, y_test, c=c)
    f1_est_qnn = compute_estimator_QNN(X_train, y_train, X_test, y_test, 'estimator')
    f1_sam_qnn = compute_estimator_QNN(X_train, y_train, X_test, y_test, 'sampler')
    
    results_dict[i] = [f1_svc, f1_qsvc, f1_peg_qsvc, f1_est_qnn, f1_sam_qnn]
    
df = pd.DataFrame.from_dict(results_dict, orient='index')
df.to_csv('/dccstor/boseukb/Q/ML/v2/BrCa_results_comparison_10PCs_v2.csv', 
          index=False, header=['SVC', 'QSVC', 'PEG_QSVC', 'EST_QNN', 'SAM_QNN'])

            



