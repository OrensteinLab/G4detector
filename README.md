# G4detector

We present G4detector, a multi-kernel convolutional neural networks, aimed at claasifiyng DNA sequences for having the potential to form G-quadroplex.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The model is implemented with Keras 2.1.6 using Tensorflow backend


### How does it work

Follow the instruction bellow:

cd path/to/G4detector/directory

If you choose to do 10-fold cross validation such as the one that was done ion the original paper:
python G4detector.py cross-val path-to-positive-set.csv path-to-negative-set.csv

If you choose to train a model on the complete dataset:
python G4detector.py train path-to-positive-set.csv path-to-negative-set.csv

If you choose to test an existing model:
python G4detector.py test path-to-dataset.csv path-to-model.h5

