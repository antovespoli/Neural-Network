# Neural-Network
Regression model for predicting value of houses

# Introduction to ML - Coursework 2

# Part 1:
Our part 1 implementation uses numpy and pickle libraries.   
We have an example_main function that uses the iris.dat dataset and creates a multi layer neural network on the data . It prints the train loss, validation loss and validation accuracy for this neural network model.   
This can be simply run by doing:  
python3 part1_nn_lib.py 

The module can also be imported and contains the following classes:
1. Layer - abstract class that other classes inherit from
2. MSELossLayer - computes mean squared error
3. CrossEntropyLossLayer - Computes the softmax followed by the negative log-likelihood loss
4. SigmoidLayer - Applies sigmoid function elementwise. Functions:   
(a) \_\_init\_\_ - constructor  
(b) forward - takes an input dataframe and performs a forward pass through the layer  
(c) backward - performs backward pass by computing gradient of loss  
5. ReLuLayer - Applies Relu function element wise. Functions:      
(a) \_\_init\_\_ - constructor  
(b) forward - takes an input dataframe and performs a forward pass through the layer  
(c) backward - performs backward pass by computing gradient of loss  
6. Linear Layer - Performs affine transformation of input. Functions:    
(a) \_\_init\_\_ - constructor. Uses xavier_init function defined to initialize weights  
(b) forward - takes an input dataframe and performs a forward pass through the layer  
(c) backward - performs backward pass by computing gradient of loss  
(d) update_params - updates weights and biases using learning rate  
7. MultiLayerNetwork - A network consisting of stacked linear layers and activation functions. Functions:    
(a) \_\_init\_\_ - constructor. Takes input dimension and creates neural network based on the neurons and activations list provided  
(b) forward - takes an input dataframe and performs a forward pass through all the layers  
(c) backward - performs backward pass by computing gradient of loss through all the layers   
(d) update_params - updates weights and biases using learning rate for each layer  
8. Trainer - Manages training of neural network. Functions:    
(a) \_\_init\_\_ - constructor. Takes multi layer neural network, batch size, epochs, learning rates, loss function and bool for shuffle   
(b) shuffle - returns shuffled version of input  
(c) train - trains the multi layer neural network by doing the forward pass and back propagation  
(d) eval_loss - evaluates the loss function for the data  
9. Preprocessor - Applies pre-processing to input and also reverts it if needed. Functions:    
(a) \_\_init\_\_ - constructor. Gets the min and max from the dataset  
(b) apply - applies minmax normalization  
(c) revert - reverts the minmax normalization  

# Part 2
Our part 2 implementation uses the torch, pickle, numpy, pandas, sklearn and matplotlib libraries.  
We have an example_main function which takes the housing.csv dataset as input, trains the model and computes the MSE and RMSE loss for training dataset. We also have hyper parameter tuning and then we output the MSE and RMSE loss on the held out test dataset for the best model. We also add some plots using matplotlib in this function.   
This can be simply run by doing:  
python3 part2_house_value_regression.py

The module can also be imported and contains the following classes and functions:
1. Network - this class inherits from torch.nn.Module and provides the framework for the neural network. Functions:  
(a) \_\_init\_\_ - takes input_dimension, output_dimension, hidden_units and dropout_rate as input.  Creates a nueral network with 3 hidden linear layers with batch normalisation before every linear layer and a dropout layer after each. We also implement ReLU activation  
(b) forward - takes the input data and does a forward pass through the neural network using ReLU activation for the hidden layers and sigmoid at the output
2. Regressor - Main class to perform the regression on the data using a neural network   
(a) \_\_init\_\_ - takes the input dataframe, number of epochs, learning rate, hidden units and drop out rate and applies pre-processing to the data and creates the neural network using the Network class. Uses an Adam optimizer for the network   
(b) \_preprocessor - takes the input dataframe, output labels (optional) and training flag. Performs pre-processing using min-max normalization and returns PyTorch tensors  
(c) fit - Trains the model on the input data using batching for the pre-defined number of epcohs. Calculates the loss and returns that for plotting purposes  
(d) predict - Runs the input through a trained neural network and outputs the predicted house values after reverting the minmax normalization  
(e) score - computes the MSE loss and RMSE loss for a given test input by running it through the neural network and comparing the predicted output with the actual output  
3. Function - RegressorHyperParameterSearch. Performs hyper parameter tuning for number of epochs, learning rate, hidden units and dropout rate. This is done by creating neural networks for each case and evaluating their performance. Returns the best combination of the hyperparameters.
4. Function - save_regressor. Saves the best model
5. Function - load_regressor. Loads the best model
