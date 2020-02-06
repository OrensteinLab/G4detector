# G4detector

We present G4detector, a multi-kernel convolutional neural networks, aimed at claasifiyng DNA sequences for having the potential to form G-quadroplex (G4).
As part of this study we generated three novel datasets of high-throughput G4 measurements for benchmarking different computational methods for the prediction task. We used genomic coordinates as retrievedin G4-seq experiment (GEO accession numbers GSE63874). The experiment included sequences identified as G4 using different stablizers (K, PDS, K+PDS). We turned each of the three sets into a classification problemby augmenting it with a negative set. Since each negative set mayhave its drawbacks, we used three different kinds of negatives: 

              1. random: random genomic sequences
              
              2. dishuffle: randomly shuffled positives while preserving dinucleotide frequencies
              
              3. PQ: predicted G-quadruplexes in the human genome according to a regular expression: [G]^{3+}[ACGT]^{1-7}[G]^{3+}[ACGT]^{1-7}[G]^{3+}[ACGT]^{1-7}[G]^{3+}

This resulted in the formation of nine different models, all of which can be found in the models folder.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The model is implemented with Keras 2.1.6 using Tensorflow backend


### How does it work

Follow the instruction bellow:

```
cd path/to/G4detector/directory

#If you choose to do 10-fold cross validation such as the one that was done ion the original paper:
python G4detector.py cross-val path-to-positive-set.fa path-to-negative-set.fa

#If you choose to train a model on the complete dataset:
python G4detector.py train path-to-positive-set.fa path-to-negative-set.fa

#If you choose to test an existing model:
python G4detector.py test path-to-dataset.fa path-to-model.h5
```

