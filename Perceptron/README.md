# CS-6350-Fall-2023
Instructions for running my perceptron library

Execute run.sh in the directory above the 'Perceptron' folder. run.sh takes three arguments:

--type: This can be "standard", "voted" or "average" ("standard" by default). This is the perceptron variant.
--T: any positive integer (10 by default). This is the number of epochs.
--r: any positive decimal (1.0 by default). This is the learning rate.

Here are some examples that can be copied and pasted:

./run.sh standard 10 1.0 
./run.sh voted 10 1.0 
./run.sh average 10 1.0 