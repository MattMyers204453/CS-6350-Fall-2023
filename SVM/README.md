# CS-6350-Fall-2023
Instructions for running my SVM library

Execute run.sh in the directory above the 'SVM' folder. run.sh takes 6 arguments. They are all required even though some are sometimes unnecessary.

--type: This can be "p", "voted" or "average" ("p" by default). (primal, dual, kernel).
--T: any positive integer (100 by default). This is the number of epochs.
--r: any positive decimal (0.001 by default). This is the learning rate.
--a: any positive decimal (0.5 by default). This constant is involved in the learning rate.
--C: any positive decimal (500.0 / 872 by default). This is the C hyperparameter in soft SVM
--g: any positive decimal (5.0 by default). This is involved in the gaussian kernel function.

Here are some examples that can be copied and pasted:

./run.sh p 100 0.001 0.5 0.55 5.0 
./run.sh d 100 0.001 0.5 0.55 5.0  
./run.sh k 100 0.001 0.5 0.55 5.0 