# NeuralNet
Simple Neural Net in C++

Uses a Linear model with least squares error. An example of training and generating from the model for binary classification is given in main.cpp. In particular, a matrix of 0's and 1's are are passed in, with rows corresponding to each training case, and two columns. A target vector is then passed in which is the result of "Anding" the entries of each row. After 100 epochs, the following predictions were made (prior to hard-thresholding):

[      0,       1, -0.7235]
[      1,       0,  -1.676]
[      0,       0,   -1.03]
[      1,       1,  -1.369]
[      0,       0,   -1.03]
[      0,       1, -0.7235]
[      1,       0,  -1.676]

After 1000 epochs, the following predictions were made (prior to hard-thresholding):

[      1,       1,   1.486]
[      1,       1,   1.486]
[      0,       0,  -2.509]
[      1,       1,   1.486]
[      0,       0,  -2.509]
[      0,       1,  -0.474]
[      1,       0, -0.5486]

Here the first two columns are input data, and the third column is the output of the neural net prior to hard-threshold. This means that positive values correpsond to a prediction of a 1, and negative a 0. We see correct predictions made in the case of "Anding" the first two columns after 1000 epochs, but not only after 100.

This project was undertaken mainly to start to learn C++. The strcutre of the code is adapted from python starter code provided in CSC321 at U of T, by Roger Grosse.
