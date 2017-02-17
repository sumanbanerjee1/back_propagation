# back_propagation
A generic artificial neural network for the MNIST data , that can be trained by using backpropagation and different optimization algorithms. 
Given an input image (28 x 28 = 784 pixels) from the MNIST dataset, the network will be trained to classify the image into
1 of 10 classes (10 digits). The implementation supports the use of the following hyper-parameters/options :
• --lr (initial learning rate η for gradient descent based algorithms)
• --momentum (momentum to be used by momentum based algorithms)
• --num hidden (number of hidden layers - this does not include the 784 dimensional
    input layer and the 10 dimensional output layer)
• --sizes (a comma separated list for the size of each hidden layer)
• --activation (the choice of activation function - valid values are tanh/sigmoid)
• --loss (possible choices are squared error[sq] or cross entropy loss[ce])
• --opt (the optimization algorithm to be used: gd, momentum, nag, adam - the mini-batch version of these algorithms are implemented)
• --batch size (the batch size to be used - valid values are 1 and multiples of 5)
• --anneal (if true the algorithm should reduces the learning rate by 70% if at any epoch the validation loss decreases 
    and then restart that epoch)
• --save dir (the directory in which the pickled model should be saved - by model it means all the weights and biases of
    the network)
• --expt dir (the directory in which the log files will be saved - The log files contain The loss and error rate at every 100 steps
    of an epoch)
• --mnist (path to the mnist data in pickeled format)

run.sh contains the format in which 'train.py' needs to be run. Before running 'train.py' you need to run 'elastic.py' that 
generates a new set of 50,000 data points from the old mnist data by introducing elastic distortions into each image. This data
is then augmented with the previous mnist data to obtain 100000 data points on which the data points are to be trained. This
provides a better generation.
