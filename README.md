# G4detector

We present G4detector, a multi-kernel convolutional neural networks, aimed at classifiyng DNA sequences for having the potential to form G-quadroplex (G4).
As part of this study we generated novel datasets of high-throughput G4 measurements for benchmarking different computational methods for the prediction task. We used genomic coordinates as retrieved in the G4-seq experiment (GEO accession numbers GSE110582). The experiment included sequences identified as G4 using different stablizers (K and K+PDS). We turned each of the sets into a classification problemby augmenting it with a negative set. Since each negative set may have its drawbacks, we used three different kinds of negatives: 

              1. random: random genomic sequences
              
              2. dishuffle: randomly shuffled positives while preserving dinucleotide frequencies
              
              3. PQ: predicted G-quadruplexes in the human genome according to a regular expression: [G]^{3+}[ACGT]^{1-7}[G]^{3+}[ACGT]^{1-7}[G]^{3+}[ACGT]^{1-7}[G]^{3+}

This resulted in the formation of nine different models, all of which can be found in the models folder.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The model is implemented with Keras 2.1.6 using Tensorflow backend

Running of ```g4_fold.py``` also requires the installation of the ViennaRNA package. See https://www.tbi.univie.ac.at/RNA/ for installation details. 


## How does it work

### Sequence + RNA secondary Strucutre

Follow the instruction bellow:

```
cd path/to/G4detector/directory
mkdir plots ; mkdir models ; mkdir predictions ; mkdir plots_arrays ; mkdir plots_arrays/roc ; mkdir plots_arrays/pr

#Getting the necessary data for g4_fold.py
mkdir path/to/positive/_lunp/files ; cd path/to/positive/_lunp/files
RNAplfold -u 1 < path/to/positive/file.fa

cd .. ; mkdir path/to/negative/_lunp/files ; cd path/to/negative/_lunp/files
RNAplfold -u 1 < path/to/negative/file.fa

#Training the model 
#structure and sequence
python g4_fold.py -p path/to/positive/file.fa -fp path/to/positive/_lunp/files -n path/to/negative/file.fa -fn path/to/negative/_lunp/files

#Predicting from an existing model
python g4 -p path/to/positive/file.fa -fp path/to/positive/_lunp/files -n path/to/negative/file.fa -fn path/to/negative/_lunp/files -mdl path/to/model/directory -nt negative_type
```

### Sequence only
Follow the instruction bellow:

```
cd path/to/G4detector/directory
mkdir plots ; mkdir models ; mkdir predictions ; mkdir plots_arrays ; mkdir plots_arrays/roc ; mkdir plots_arrays/pr

#Training the model
#sequence only
python g4.py -p path/to/positive/file.fa -n path/to/negative/file.fa 
```
