# Quantum Deep Learning

Example Jupyter notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker.

## :books: Background

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

The [SageMaker example notebooks](https://sagemaker-examples.readthedocs.io/en/latest/) are Jupyter notebooks that demonstrate the usage of Amazon SageMaker.

Quantum machine learning (QML) has emerged as the 
potential solution to address the challenge of handling an 
ever-increasing amount of data. With these advancements 
there is potential for reduced training times, and hybrid 
quantum-classical architectures.

## :hammer_and_wrench: Setup

The quickest setup to run example notebooks includes:
- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

## :notebook: Examples

### Introduction to Seqeuntial Circuits

These examples provide quick walkthroughs to get you up and running with the labeling job workflow for Amazon SageMaker Ground Truth.

- [***Start Here***](/0_introduction_to_sequential_circuits.ipynb) is an end-to-end example that shows how to build sequential circuits that are optimized for circuit training.

### Training Sequential Circuits

These examples provide several examples how variational circuits can be trained.

- [***Training with Pennylane and Numpy***](1_training_circuits_numpy.ipynb) 
 This notebook shows how to use Pennylane's optimizers for circuit training, with Numpy to facilitate training.

- [***Training with PyTorch***](2_training_circuits_pytorch.ipynb) 
 This notebook shows how to wrap circuits into a Pytorch layer, and model that can be trained.
 
 - [***Training with Tensorflow***](3_training_circuits_tensorflow.ipynb) 
 This notebook shows how to wrap circuits into a Tensorflow layer, and model that can be trained.
 This method is currently limited by the issue stated here: https://github.com/PennyLaneAI/pennylane/issues/937.
 The implication is that only circuit structures that have two expectation values can be trained.

## :balance_scale: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/master/LICENSE.txt) file.

## :handshake: Contributing

Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed.
Please read our [contributing guidelines](https://github.com/aws/amazon-sagemaker-examples/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.

