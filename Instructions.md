# Quantum CNN

## Introduction

This blog post is about utilizing AWS ML services to build Quantum Convolutional Neural Networks (QCNN). To do so we utilize Amazon SageMaker, PennyLane, and PyTorch to train and test, QCNNs on simulated quantum devices. Previously it has been demonstrated that QCNNs can be used for binary classification tasks. This blog post will go one step further and show how to construct a QCNN for multi-class classification and apply it to image classification tasks. Based on the steps shown in this post, you can begin to explore the use cases of QCNNs by building, training, and evaluating your own model. 

## Approach

To implement this solution, we used PennyLane, PyTorch, SageMaker notebooks, and the public MNIST, and Fashion-MNIST datasets. The specific notebook instances were ml.m5.2xlarge - ml.m5.24xlarge.

### PennyLane

PennyLane is a Python library for programming quantum computers. Its differentiable programming paradigm enables the execution and optimization of quantum programs on a variety of simulated and hardware quantum devices. It manages the execution of quantum computations, to include the evaluation of circuits and their gradients. This information can then be forwarded to the classical framework, creating seamless quantum-classical pipelines for applications to include the building of Quantum Neural Networks.Additionally PennyLane offers integration with ML frameworks such as PyTorch.

### PyTorch

PyTorch is an open-source, deep learning framework that makes it easy to develop ML models and deploy them to production. PyTorch has been integrated into the PennyLane’s ecosystem allowing for the creation of hybrid Quantum-Classical ML models. Furthermore this integration has allowed for the native training of QML models in SageMaker.

### Amazon SageMaker

SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy ML models quickly. SageMaker removes the heavy lifting from each step of the ML process to make it easier to develop high-quality models. With its integration of PyTorch, practitioners are able to build, train, and deploy their own PyTorch models on SageMaker.

## Overview of Quantum Computing

#### Quantum Bits

Quantum computing isn’t built on bits that arent binary in nature. Rather quantum bits (qubits) are two-state quantum mechanical systems that can be a combination of both zero and one at the same time. In addition to this property of “superposition”, qubits can be “entangled” allowing for N qubits to act as a group rather than in isolation. As a result qubits can achieve exponentially higher information density (2^N) than the information density of a classical computer (N). 

[Image: image.png]

#### Quantum Circuits:

Quantum circuits are the underlying framework for quantum computation. Each circuit is built sequentially and consist of three stages: Preparation of a fixed initial state, gate transformations, and measurement of an observable(s). 

In the first stage the qubits are initialized with the quantum or classical data that is going to be used for computation. Several options exist for encoding classical data, on to qubits. For our experiment we elected to use Amplitude embedding, which takes 2^qubits data points as the initial state. 

In the second stage, gate based transforms are applied to the qubits. This stage can be as simple as a one gate operation or as complex as a grouping of gates. When several gates are grouped they are often referred to as a unitary. 

Lastly the circuit is completed with the measurement of an observable. This observable may be made up from local observable for each wire in the circuit, or just a subset of wires. Prior to this stage the qubits have been in superposition, again representing a mixture of the states 0 and 1. However when a qubit is measured, this state collapses into either 0 or 1, with an associated probability of doing so. 

#### Quantum Optimization

Variational or parametrized quantum circuits are quantum algorithms that depend on free parameters. Much like standard quantum circuits they consists of: the preparation of the fixed input state, a set of unitaries parameterized by a set of free parameters θ, and measurement of an observable ^B at the output. The output or expectation value with some classical post-processing can represent the cost of a given task.  Given this cost the free parameters θ=(θ1,θ2,...) of the circuit can be tuned and optimized. This optimization can leverage a classical optimization algorithm such as stochastic gradient descent or adam.

#### Quantum Convolutional Neural Network

Quantum convolutions are a unitary that act on neighboring pairs of cubits. When this unitary is applied to every neighboring pair of cubits a convolution layer is formed. This convolution layer mirrors the kernel based convolutional layer found in the classical case. Below is the structure of the convolutional unitary that was used for this project.



[Image: conv.png]
These convolutions are followed by pooling layers, which are effected by measuring a subset of the qubits, and using the measurement results to control subsequent operations. Shown below is the structure that was used:



[Image: image]


The analogue of a fully-connected layer is a multi-qubit operation on the remaining qubits before the final measurement. This essentially maps the information from the remaining qubits on the ancillary qubits for our final measurement. The structure for doing so is the following:

[Image: index.png]

The resulting network architecture is two convolution layers, two pooling layers, and the ancillary mapping.

[Image: QCNN.png]




### Building the model

#### PennyLane Basics

1. To begin constructing a quantum circuit we must first define the quantum simulator, or hardware device that will be used. 

* Note: Qubits are often reffered to as wires.

```
import pennylane as qml
dev = qml.device('default.qubit', wires=1) # Specify how many qubits are needed
```

1. Now that the device is defined, we can begin building or quantum circuit. 

* Note: the [qnode](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QNode.html) decorator attaches the circuit to the quantum device for computation and can also specify the gradient method that will be used.

```
@qml.qnode(dev) # qnode decorator
def circuit(input):
  qml.RX(input, wires=0) # Quantum Gate 
  return qml.sample(qml.PauliZ(wires=0)) # Measurement of the PauliZ 
```

1. To visualize the circuit we can call draw_mpl(), with the required inputs of the circuit.

```
from matplotlib import pyplot as plt
input = 1
fig, ax = qml.draw_mpl(circuit)(input)
plt.show()
```


[Image: simple circuit.png]

#### Advanced PennyLane

* Note about how PennyLane was expanded to support sequential circuit building.
* Link to example notebook [?]

First we will establish the quantum device that will be used for our circuit.

```
import pennylane as qml

# First we will specify how many qubits will be used in our model
# Note: Qubits are often refferred to as wires.
num_wires = 12

# Since we are doing multiclass classification with 4 classes, 
# We will need 4 ancillary wires.
num_ancilla = 4

# Again, we will specify the quantum device used for computation
# We elected to use PennyLane's lighting qubit simulator.
device_type = 'lightning.qubit'

# Lastly we will intialize the device, with the number of wires needed.
dev = qml.device(device_type, wires=num_wires) 
```

 Next we will define the quantum model.

```
def circuit(num_wires, num_ancilla):

    # The qnode decartor will attach our circuit to the device
    # previously specified.
    
    # Note: That the interface specifies that it will be used 
    # inside of a Pytorch model.
    
    # Note: The differentiation method is specified as adjoint.
    @qml.qnode(dev, interface='torch', diff_method ='adjoint')
    
    # This inner function is resposible for mapping, inputs
    # and a list of all the parameters to the circuit. 
    def func(inputs, params):
    
        # We will create a list of work wires, this will be
        # where all work of the model will be accomplished.
        work_wires = list(range(num_wires - num_ancilla))
        
        # We will also create a list of ancillary wires where the
        # the final measurements will be achomplished.
        ancilla_wires = list(range(len(work_wires),num_wires))

        # We are using the AmplitudeEmbedding provided by Pennylane
        # to map the image onto our quantum circuit.
        # This is the same as speciying the input any a traditional ML
        # model.
        # This is only done on the work wires as this is where
        # the processing will happen. 
        
        AmplitudeEmbedding(inputs , wires=work_wires , normalize=True)

        # Next we will specify our "hidden layers".
        
        # Each layer is inialized with two arguments: 
        # the convulutanal unitary that has been constructed, 
        # and the number of free parameters that it needs.
        
        # This is then called, with the work_wires that it will be applied
        # to and the parameters that will be used.
        
        # This layer will return all the work wires, and the remaining
        # parameters not used by this layer.
        work_wires, params = Conv1DLayer(unitarity_conv1d, 15)(work_wires, 
              
        # This layer will return a subset of the the work wires, 
        # and the remaining parameters not used by this layer.                                                                                                                                                                                                                 params)
        work_wires, params = PoolingLayer(unitarity_pool, 2)(work_wires,
                                                               params)
                                                               
        work_wires, params = Conv1DLayer(unitarity_conv1d, 15)(work_wires,
                                                                params)
        work_wires, params = PoolingLayer(unitarity_pool, 2)(work_wires,
                                                                 params)
   
        #We will then map the outputs to anicallary wires.
        unitarity_toffoli(work_wires,ancilla_wires)

        #Lastly we will measure observable PauliZ, on each of the
        # ancilla wires. PauliZ will measure the expectation value
        # for all the information in the Z plane.
        return [qml.expval(qml.PauliZ(wire)) for wire in ancilla_wires]
    return func
```

It is now time to wrap our quantum circuit into a PyTorch Model.

```
# First we will specify a dictionary for our parameters
# for simplicity we will call it params.
# the second value represent the number of parameters needed.
# For each conv1d layer you will need unitary_num_params * num_work_wires
# For each pooling layer you need unitary_num_params * num_work_wires
params_shapes = {"params": 357} 
 
# Now we call qml.qnn.TorchLayer, passing in the outer circuit, 
# with the number of wires, and number of ancilla wires.
# We will also pass the parameters dictionary, and the initialization 
# method for those parameters, in which we used a normal distrobution.  
qlayer = qml.qnn.TorchLayer(circuit(num_wires, num_ancilla), 
                            params_shapes,torch.nn.init.normal_)
 
# Lastly we will specify a softmax layer to classical process our
# quantum output.
output = torch.nn.Softmax(dim=1)
```

Now we will initialize the model with the layers that we just made.

```
#Defining the model
model = torch.nn.Sequential(qlayer,output)
```


[code snippets [Dan Ford](https://quip-amazon.com/VFd9EABrUlJ) can take a look at how to do this]. [Perhaps we can show how we modified PennyLane’s code and link to it]‘

### Training Quantum Model

For training we conducted an experiment on a subset of the MNIST, and Fashion-MNIST data sets.

```
# Hyperparameters
epochs = 8
batch_size = 32
train_samples = len(train_data)
batches = train_samples // batch_size
```



```
from tqdm.notebook import trange

opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    
    running_loss = 0
    batch = 0
    model.train()
    
    for batch, i in zip(train_loader,trange(batches)):
        data = batch[0]
        target = batch[1]      
        opt.zero_grad()
        pred = model(data)
        loss_evaluated = loss(pred, target)
        loss_evaluated.backward()
        opt.step()
        running_loss += loss_evaluated
        print(running_loss.item() / (i + 1), end ='\r')
        
    avg_loss = running_loss / batches
    res = [epoch + 1,avg_loss]
    print("Epoch: {:2d} | Loss: {:3f} ".format(*res))
```

#### Evaluating the Model

```
# Since we're not training, we don't need to calculate the 
# gradients for our outputs
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        data = batch[0]
        target = batch[1]
        
        predicted = model(data)
        
        predicted = torch.argmax(predicted, 1)

        total += target.size(0)
        
        correct += (predicted == target).sum().item()

print('Accuracy of the network on {:2d} test images: {:.3f} %'.format(total,(100 * correct / total)))
```

#### Saving Model

```
torch.save(model.state_dict(), PATH)
```

#### Loading Model

```
model = torch.nn.Sequential(qlayer,output)
model.load_state_dict(torch.load(PATH))
model.eval()
```

### Deploy Model to SageMaker Endpoint

To deploy this model to a SageMaker endpoint one would follow the standard steps for deploying a PyTorch model on SageMaker. The only additional step needed would be to extend the existing PyTorch container instance to include PennyLane.

### Results

For the subset of the MNIST data set the average accuracy after 8 epochs was 92%.
For the Fashion-MNIST data set the average accuracy after 8 epochs was 88%

## Next steps: 

* There are various different methods that can be used to encode data on to the quantum circuit. To be able to further leverage the information density of qubits these methods should be further explored. This could allow for larger and more complex datasets to be used for training. 
* Exploring further, the specific convolutional unitary used in this blog is one of many that could be used. It is possible that there are more efficient methods for conducting the convolution.
* Lastly due to the quantum circuits being supported in PyTorch it is possible to explore Quantum-Classical model architectures for multi-class classification. Which enables support of transfer learning from well established pre-built models.

## Conclusion:

In this post, we demonstrated how quantum computing can be leveraged for multi-class image classification tasks. Moreover we showed you how this can be accomplished on available Amazon SageMaker instances, using PennyLane for building QCNNs and PyTorch to facilitate model training and deployment.  Furthermore, we discussed mechanisms that enabled the construction of QCNNs, and their training. 

## About the authors

Zach Benson is an Undergraduate Data Science Student at the United States Air Force Academy.
 Zach joined AWS the past two summers through [Amazon Skillbridge Fellowship](https://amazon.jobs/en/landing_pages/mil-transition). 
