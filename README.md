# Quantum CNN — Multi-Class Image Classification

**Extending PennyLane for multi-class quantum ML | AWS BRIDGE Fellowship 2022**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-blueviolet)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**Extended PennyLane's quantum ML library to enable multi-class image classification — achieving 92% accuracy on MNIST (4-class) and 88% on Fashion-MNIST, surpassing existing binary-only QCNN benchmarks.**

## The Problem

Prior Quantum Convolutional Neural Networks (QCNNs) were architecturally limited to binary classification tasks. Multi-class capability required extending the framework itself — not just tuning hyperparameters — representing a fundamental gap in the open-source quantum ML ecosystem. This project filled that gap.

## The Contribution

- Extended PennyLane's library architecture to support multi-class output via ancillary qubit mapping
- Integrated PyTorch into the PennyLane ecosystem for hybrid quantum-classical training (`qml.qnn.TorchLayer`)
- Built and evaluated an end-to-end training pipeline on AWS SageMaker (ml.m5.2xlarge — ml.m5.24xlarge instances)
- Benchmarked against MNIST (4-class subset) and Fashion-MNIST datasets
- Demonstrated multi-class quantum ML is viable beyond binary classification

## Results

| Dataset | Classes | Accuracy |
|---|---|---|
| MNIST (4-class) | 4 | **92%** |
| Fashion-MNIST | 4 | **88%** |
| Prior QCNN baseline (binary) | 2 | ~85% (reference) |

Training: 8 epochs, batch size 32, Adam optimizer (lr=0.001).

## Architecture Overview

The model uses **12 qubits** — 8 work qubits and 4 ancillary qubits (one per output class). Input images are encoded onto work qubits via **amplitude embedding**, which maps 2^n data points onto n qubits.

The circuit passes through two rounds of **convolution → pooling** layers. Each convolutional unitary operates on neighboring qubit pairs using parameterized U3 gates, CNOT entanglement, and rotation gates (15 parameters per unitary). Pooling layers reduce the active qubit count by applying controlled rotations and tracing out half the qubits.

After pooling, a **Toffoli-based mapping** transfers information from the remaining work qubits onto the 4 ancillary qubits. PauliZ expectation values are measured on each ancilla, producing a 4-element output vector fed through a classical softmax layer. The full model has **357 trainable parameters** and runs on PennyLane's `lightning.qubit` simulator.

The quantum circuit is wrapped as a `TorchLayer`, making it a drop-in component within a standard `torch.nn.Sequential` model — enabling native PyTorch training and gradient computation via the adjoint differentiation method.

## Tech Stack

- **Python 3.x**
- **PennyLane** — quantum ML framework, extended for multi-class support
- **PyTorch** — classical ML integration layer via `qml.qnn.TorchLayer`
- **AWS SageMaker** — training infrastructure (notebook instances)
- **Datasets**: MNIST (4-class subset), Fashion-MNIST

## Getting Started

### Dependencies

```bash
pip install pennylane pennylane-lightning torch torchvision numpy matplotlib tqdm
```

### Notebooks

| Notebook | Description |
|---|---|
| `0_introduction_to_sequential_circuits.ipynb` | Intro to quantum circuit construction and the layer abstraction |
| `1_training_circuits_numpy.ipynb` | Training variational circuits with PennyLane + NumPy |
| `2_0_training_circuits_pytorch_MNIST.ipynb` | Multi-class QCNN on MNIST (4-class) — primary result |
| `2_1_training_circuits_pytorch-FMNIST.ipynb` | Multi-class QCNN on Fashion-MNIST |
| `2_2_training_circuits_pytorch-CIFAR.ipynb` | QCNN on CIFAR dataset |
| `3_training_circuits_tensorflow.ipynb` | TensorFlow integration (limited by PennyLane issue [#937](https://github.com/PennyLaneAI/pennylane/issues/937)) |

### Running

```bash
jupyter notebook 2_0_training_circuits_pytorch_MNIST.ipynb
```

The model definition is in `model.py` — contains the Conv1D, Pooling, and Toffoli layer classes. All training runs on PennyLane's `lightning.qubit` simulator (no quantum hardware required).

## Context

- Built during **Amazon Web Services BRIDGE Fellowship**, ML Solutions Lab — Summer 2022
- Presented to AWS Emerging Technologies team

## Author

Built by Zachary Benson | AWS BRIDGE Fellow | Space Force Officer
[LinkedIn](https://linkedin.com/in/zacharybenson)
