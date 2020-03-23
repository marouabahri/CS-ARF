# CS-ARF
Repository for the Compressed Sensing Adaptive Random Forest (CS-ARF) algorithm implemented in MOA.

For more informations about MOA, check out the official website: 
http://moa.cms.waikato.ac.nz

## Citing CS-AdaptiveRandomForest
To cite the CS-ARF in a publication, please cite the following paper: 

> Maroua Bahri, Heitor Murilo Gomes, Albert Bifet, Silviu Maniu.
> CS-ARF: Compressed Adaptive Random Forests for Evolving Data Stream Classification. In the International Joint Conference on Neural Networks (IJCNN), 2020.

## Important source files
The implementation and related codes used in this work are the following: 
* CS-AdaptiveRandomForest.java: the compressed sensing AdaptiveRandomForest using the several random projections, one for each Tree in the ensemble.

## How to execute it
To test the CS-kNN, you can copy and paste the following command in the interface (right click the configuration text edit and select "Enter configuration”).
Sample command with 30 trees and reduce the dimensionality to 10: 

`EvaluatePrequential -l (meta.CS_AdaptiveRandomForest -l (CS_ARFHoeffdingTree -a 10) -s 30) -s (ArffFileStream -f /pathto/tweet500.arff) -e BasicClassificationPerformanceEvaluator`

Explanation: this command executes CS-kNN prequential evaluation precising the output and input dimensionality, d and f respectively on the tweet500 dataset (-f tweet1.arff). 
**Make sure to extract the tweet500.arff dataset, and setting -f to its location (pathto), before executing the command.**

## Datasets used in the original paper
The real datasets are compressed and available at the root directory. 
