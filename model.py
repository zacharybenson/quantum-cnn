# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""This module contains code related to Braket-Pennylane integration,
   for training quantum neural networks.
"""

from abc import ABC, abstractmethod, abstractproperty
import pennylane as qml
        
def unitarity_conv1d(wires, params): 
    """Unitarity - Convulution Base
       Number of Params: 15
    """
    # Defines the structure of the Convulution Unitarity
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])
                        
def unitarity_pool(wires, params):
    """Unitarity - Pooling Base
       Number of Params: 2  
    """
    # Defines the structure of the Pooling Unitarity
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]]) 
    

def unitarity_toffoli(work_wires,ancilla_wires):
    
    num_work_wires = len(work_wires)
    num_ancilla_wires = len(ancilla_wires)

    for i,j in zip(range(0,num_work_wires, 2), range(0,num_ancilla_wires,4)):
       
        # Set up
        subset = [work_wires[k] for k in range(i,i+2)]
        subset.append(ancilla_wires[j])
        toffoli_wires = subset
        
        # Gate 1
        qml.Toffoli(wires=toffoli_wires)
        qml.PauliX(toffoli_wires[0])
        toffoli_wires[-1] += 1
        
        # Gate 2
        qml.Toffoli(wires=toffoli_wires)
        qml.PauliX(toffoli_wires[0])
        qml.PauliX(toffoli_wires[1])
        toffoli_wires[-1] += 1
        
        # Gate 3
        qml.Toffoli(wires=toffoli_wires)
        qml.PauliX(toffoli_wires[0])
        toffoli_wires[-1] += 1

        # Gate 4
        qml.Toffoli(wires=toffoli_wires)
        
        
class AbstractLayerClass(ABC):
    """Unitarity Abstract Layer"""

    def __init__(self, unitarity, unitarity_num_params):
        """Creates a new layer and returns active wires, and parameters not used."""
        self.unitarity = unitarity
        self.unitarity_num_params = unitarity_num_params 
        self.unitarities = []  
  
    @abstractmethod
    def structure(self, wires):
        """Defines the structure of the Abstract layer"""
        raise NotImplementedError
    
    @abstractmethod
    def list_wires(self):
        """Defines the active wires of the circuit"""
        raise NotImplementedError
    
    def params(self):
        """Maps the params of the Unitarities, to the layer"""
        params = []
        for unitarity in self.unitarities:
            params.append(unitarity.params)
        return params
    
    def set_params(self, params):
        """Set the params of the Unitarities in the layer"""
        start_index = 0
        for unitarity in self.unitarities:
            unitarity.params = params[start_index : start_index + self.unitarity_num_params]
            start_index = start_index + self.unitarity_num_params
            
    def count_params(self):
        """Counts the params of the Unitarities in the layer"""
        num_params = 0
        for unitarity in self.unitarities:
            num_params = num_params + self.unitarity_num_params
        return num_params

    def __call__(self, wires, params):
        #Building the Circuit
        self.wires = wires
        self.num_wires = len(wires)
        self.structure(params)
        self.num_params = self.count_params()
        
        #Computing Active Wires and Active Params
        active_wires = self.list_wires()
        active_params = params[self.num_params :]
        return [active_wires, active_params]
            
class Conv1DLayer(AbstractLayerClass):
    """Convulution Layer
       Number of Params:  15 * x, where x ≡ number of qubits
    """

#     def __init__(self, unitarity_num_params):
#         """Creates a new Conv1D layer and returns active wires, and parameters not used."""
#         super().__init__(unitarity_conv1d, unitarity_num_params)
    
    def structure(self , params):
        """Defines the structure of the Convulution Layer"""
        for i in range(0,self.num_wires - 1, 1):
            self.unitarities.append(self.unitarity([self.wires[i] , self.wires[i + 1]], 
                                                   params[self.unitarity_num_params * i: (self.unitarity_num_params * i) + self.unitarity_num_params]))
        self.unitarities.append(self.unitarity([self.wires[0], self.wires[self.num_wires - 1]], params[15 * (self.num_wires - 1): 15 * (self.num_wires - 1) + 15]))
            
    def list_wires(self):
        """Returns all wires"""
        return self.wires
    
class PoolingLayer(AbstractLayerClass):
    """Pooling Layer
       Number of Params: 2 * x , where x ≡ number of qubits
    """

#     def __init__(self, unitarity_num_params):
#         """Creates a new Pooling layer and returns active wires, and parameters not used."""
#         super().__init__(unitarity_pool, unitarity_num_params)

    def structure(self, params):
        """Defines the structure of the Pooling Layer"""
        for i in range(0,self.num_wires - 1, 2): #FIX ME!
            self.unitarities.append(self.unitarity([self.wires[i] , self.wires[i+1]] , 
                                                   params[i * self.unitarity_num_params : (i * self.unitarity_num_params) + self.unitarity_num_params]))  
        self.unitarities.append(self.unitarity([self.wires[0] , self.wires[self.num_wires-1]] , 
                                                params[2 * (self.num_wires - 1): 2 * (self.num_wires - 1) + 2]))
                          
    def list_wires(self):
        """Returns every other wire"""
        return self.wires[1::2]
    