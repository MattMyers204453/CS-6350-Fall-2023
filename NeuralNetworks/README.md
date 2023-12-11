# CS-6350-Fall-2023
Instructions for running my SVM library

Execute run.sh in the directory above the 'SVM' folder. run.sh takes 6 arguments. They are all required even though some are sometimes unnecessary.

--T: any positive integer (10 is best). This is the number of epochs.
--r: any positive decimal (0.01 is best). This is the learning rate.
--d: any positive decimal (0.3 is best). This constant is involved in the learning rate.
--width: any positive decimal (50 is best). This is the width of the neural network

Here is an example that can be copied and pasted:

./run.sh 10 0.01 0.3 0.55 50