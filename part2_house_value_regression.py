import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt

# avoid interupting execution when plotting
plt.ion()

class Network(torch.nn.Module):
    
    def __init__(self, input_dimension, output_dimension, hidden_units=32, dropout_rate=0.3):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden_units)
        self.layer_2 = torch.nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = torch.nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.batch_norm1 = torch.nn.BatchNorm1d(input_dimension)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_units)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_units)
        self.batch_norm4 = torch.nn.BatchNorm1d(hidden_units)
        self.drops = torch.nn.Dropout(dropout_rate)
        self.output_layer = torch.nn.Linear(in_features=hidden_units, out_features=output_dimension)

    def forward(self, input_data):
        x = self.batch_norm1(input_data)
        x = torch.nn.functional.relu(self.layer_1(x))
        x = self.drops(x)
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.drops(x)
        x = self.batch_norm3(x)
        x = torch.nn.functional.relu(self.layer_3(x))
        x = self.drops(x)
        x = self.batch_norm4(x)
        output = torch.sigmoid(self.output_layer(x))
        return output


class Regressor():

    def __init__(self, x, nb_epoch=100, learning_rate=0.0002, hidden_units=32, dropout_rate=0.3):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.col_means = None
        self.max_values = None
        self.min_values = None
        self.y_max_values = None
        self.y_min_values = None

        self.ocean_prox_binarizer = None

        self.knnimputer = None

        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        self.model = Network(self.input_size, self.output_size, hidden_units, dropout_rate=0.3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own

        if not training:
        	# check comments in else section to understand each line
            
            #x = x.fillna(self.col_means)
            

        	# transforming the ocean proximity thx to label binarizer fitted during training
            ocean_prox = self.ocean_prox_binarizer.transform(x["ocean_proximity"])

            # concatenating data
            ocean_cols = list(self.ocean_prox_binarizer.classes_)
            ocean_df = pd.DataFrame(ocean_prox)
            ocean_df.columns = ocean_cols
            ocean_df.index = x.index
            # add the dataframe to x
            x = pd.concat([x, ocean_df], axis=1)
            
            x = x.drop(["ocean_proximity"], axis = 1)
            normalized_x = (x - self.min_values) / (self.max_values - self.min_values)
            normalized_x = pd.DataFrame(self.knnimputer.transform(normalized_x))
            x = torch.tensor(normalized_x.values, dtype=torch.float32)
            if isinstance(y, pd.DataFrame):
        	    normalized_y = (y - self.y_min_values) / (self.y_max_values - self.y_min_values)
        	    y_tensor = torch.tensor(normalized_y.values, dtype=torch.float32)
        else:
            # removing outliers from training only

            knn = KNNImputer(n_neighbors=5, add_indicator=True, missing_values=np.nan)

            # Use this command to print all columns and rows in the dataframe
            # pd.set_option("display.max_rows", None, "display.max_columns", None)

            # get means of all columns and replace NaN values with the mean of the corresponding column
            # is this a good way to handle NaN values or should we do something else?
            """
            self.col_means = x.mean(skipna=True)
            x = x.fillna(self.col_means)
            """

            # for the ocean_proximity column we need to use a binarizer from sklearn
            lb = preprocessing.LabelBinarizer()
            # what to do if ocean proximity has a NaN value?
            ocean_prox = lb.fit_transform(x["ocean_proximity"])
            self.ocean_prox_binarizer = lb
            # create column names: ocean_cols = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
            ocean_cols = list(lb.classes_)
            # create a dataframe with the new columns
            ocean_df = pd.DataFrame(ocean_prox)
            ocean_df.columns = ocean_cols
            ocean_df.index = x.index
            # add the dataframe to x
            x = pd.concat([x, ocean_df], axis=1)
            # remove the ocean_proximity column from x so it does not have categoical values
            x = x.drop(["ocean_proximity"], axis = 1)
            
            # collect the max and min values of each column for use on validation set later
            self.max_values = x.max()
            self.min_values = x.min()

            # normalize dataset by implementing min max scaling on the each column of the dataset so values are between 0 and 1
            normalized_x = (x - x.min()) / (x.max() - x.min())

            knn.fit(x)

            self.knnimputer = knn

            normalized_x = pd.DataFrame(knn.transform(normalized_x))

            # convert x to tensor
            x = torch.tensor(normalized_x.values, dtype=torch.float32)

            if isinstance(y, pd.DataFrame):
                # normalize y and convert to tensor
                self.y_max_values = y.max()
                self.y_min_values = y.min()
                normalized_y = (y - y.min()) / (y.max() - y.min())
                y_tensor = torch.tensor(normalized_y.values, dtype=torch.float32)

        # Return preprocessed x and y, return None for y if it was None
        return x, (y_tensor if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, x_val=None, y_val=None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        verbose = False

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        has_validate = False
        if x_val is not None and y_val is not None:
            has_validate = True
            X_val, Y_val = self._preprocessor(x_val, y = y_val, training = False) # Do not forget

        verbose and print("Initial X : ", X, sum(torch.isnan(X)))
        verbose and print("Initial Y : ", Y, sum(torch.isnan(Y)))
        data_size, input_size = X.shape
        verbose and print("Sizes : ", data_size, input_size)
        # should become a parameter
        minibatch_size = 128

        # define loss function as MSE loss
        loss_fn = torch.nn.MSELoss()
        val_loss_fn = torch.nn.MSELoss()

        # for plotting purpose
        losses = []
        val_losses = []

        # run the loop for number of epochs given 
        for epoch in range(self.nb_epoch):

            # make sure our batch are randomly sampled and change at each epoch
            permutation = torch.randperm(data_size)
            verbose and print("Permutation : ", permutation)

            epoch_losses = []

            # get through data with batches
            for i in range(0, data_size, minibatch_size):
                
                # prepare optimizer
                self.optimizer.zero_grad()

                # get batch indices
                indices = permutation[i:i + minibatch_size]
                X_batch, Y_batch = X[indices], Y[indices]
                verbose and print("X batch : ", X_batch, sum(torch.isnan(X_batch)))
                verbose and print("Y batch : ", Y_batch, sum(torch.isnan(Y_batch)))

                # get the predictions
                Y_pred = self.model.forward(X_batch.float())
                verbose and print("Y pred : ", Y_pred, sum(torch.isnan(Y_pred)))

                # get the loss
                loss = loss_fn(Y_pred, Y_batch.float())
                verbose and print("Loss : ", loss, torch.isnan(loss))
                epoch_losses.append(loss.item())
                #if epoch == 0 and i == 0:
                    #losses.append(loss.item())

                # do back propagation
                loss.backward()
                self.optimizer.step()

            # end of an epoch - print the loss
            losses.append(np.mean(epoch_losses))
            verbose and print("Epoch:", epoch, "\tLoss: ", loss.item())

            # validation score
            if has_validate:
                with torch.no_grad():
                    Y_val_pred = self.model.forward(X_val.float())
                    val_loss = val_loss_fn(Y_val_pred, Y_val.float())
                    val_losses.append(val_loss)

        #return self
        return losses, val_losses

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget

        # run the input through the NN
        predicted_y = self.model.forward(X.float())

        # convert output tensor to dataframe
        predicted_y = predicted_y.detach().numpy()
       	predicted_y = pd.DataFrame(predicted_y, columns=["median_house_value"])

       	# revert the min-max normalization normalization performed on Y
       	multiplication_constant = self.y_max_values - self.y_min_values
       	mult_result = pd.DataFrame(predicted_y.values * multiplication_constant.values, columns=predicted_y.columns, index=predicted_y.index)
        scaled_predicted_y = mult_result + self.y_min_values

        # convert to numpy array
        scaled_predicted_y = scaled_predicted_y.values

        return scaled_predicted_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # define loss function as MSE loss
        loss_fn = torch.nn.MSELoss()

        # preprocess data - no Manon, it's already preprocessed in predict (as you imposed)
        #_, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        # predict Y - return a array - use x as it will be preprocessed inside predict function (imposed)
        Y_pred = self.predict(x)
        Y_pred = torch.tensor(Y_pred)

        Y = torch.tensor(y.values, dtype=torch.float32)

        loss = loss_fn(Y_pred, Y).item()

        return loss, loss**(1/2)  # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(dataset, output_label, k=5, seed=10): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # define the hyper parameters space
    HYPER_PARAMS_SPACE = []
    for nb_epoch in [200]:
        for lr in [0.002, 0.001]:
            for hidden_units in [48, 64, 80]:
                for dropout_rate in [0.3, 0.4, 0.5]:
                    HYPER_PARAMS_SPACE += [[nb_epoch, lr, hidden_units, dropout_rate]]

    print(HYPER_PARAMS_SPACE)

    # copy to be safe with data shuffling
    data = dataset.copy()
	
    # shuffle the dataset first
    data.sample(frac=1)

    # store performances
    avg_RMSE_s = np.zeros((len(HYPER_PARAMS_SPACE), 1))

    N, _ = data.shape
    a = N // k

    # last batch will be bigger than the others by (N % k, which is < k by definition)
    for i, hyper_params in enumerate(HYPER_PARAMS_SPACE):

        # for each hyper_params tuple, do a cross validation to estimate performance
        for j in range(0, k):

            # split w/ train and validate
            validate = data[a*j:a*(j+1)]
            train = pd.concat((data[:a*j], data[a*(j+1):]), axis=0)
        
            # Spliting input and output
            x_train = train.loc[:, train.columns != output_label]
            y_train = train.loc[:, [output_label]]

            x_val = validate.loc[:, validate.columns != output_label]
            y_val = validate.loc[:, [output_label]]

            # define the regressor with the hyperparameters and the data
            regressor = Regressor(x_train, nb_epoch = hyper_params[0], learning_rate=hyper_params[1])
            losses = regressor.fit(x_train, y_train)

            #_, RMSE = regressor.score(x_train, y_train)
            _, RMSE = regressor.score(x_val, y_val)

            if j == 0:
                avg_RMSE_s[i] = RMSE
            else: # with this formula we avoid saving the confusion matrixs
                avg_RMSE_s[i] = (j / (j + 1)) * (avg_RMSE_s[i] + (RMSE / j))

    # get the mean performance for each one
    opt_params_indice = np.argmin(avg_RMSE_s)

    # Return the chosen hyper parameters (and the associated mean performance - might delete it) 
    return HYPER_PARAMS_SPACE[opt_params_indice], avg_RMSE_s[opt_params_indice]

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs

    # Created a smaller subset of the data to test
    #data = pd.read_csv("test_housing.csv")

    data = pd.read_csv("housing.csv")

    data = data[data['population']<25000]

    # shuffle and split in train and test
    data = data.sample(frac=1).reset_index(drop=True)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    
    # Spliting input and output
    x_train = train.loc[:, train.columns != output_label]
    y_train = train.loc[:, [output_label]]

    x_test = test.loc[:, test.columns != output_label]
    y_test = test.loc[:, [output_label]]


    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    regressor = Regressor(x_train, nb_epoch=200, learning_rate=0.001, hidden_units=36, dropout_rate=0.4)
    losses, val_losses = regressor.fit(x_train, y_train, x_test, y_test)

    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='log Loss', title='Log Loss Curve of the Network')
    ax.plot(range(len(losses)), losses, color='blue')
    ax.plot(range(len(val_losses)), val_losses, color='red')
    plt.yscale('log')
    fig.savefig("figures/loss_vs_epochs.png")
    plt.show()

    # predicting with dummy data just to check
    print("Prediction Input:")
    output = regressor.predict(test.loc[:, test.columns != output_label])
    fig, ax = plt.subplots()
    ax.set(xlabel='Price prediction', ylabel='Occurences', title='Histogram of the price predictions')
    plt.hist(output, bins=40)
    plt.savefig("figures/prediction_histogram.png")
    plt.show()
    #print(output)
    
    save_regressor(regressor)

    # # Error
    print("Result on train : ")
    MSE, RMSE = regressor.score(x_train, y_train)
    print("MSE : ", MSE, "  |   RMSE : ", RMSE)
    print("Result on test : ")
    MSE, RMSE = regressor.score(x_test, y_test)
    print("MSE : ", MSE, "  |   RMSE : ", RMSE)
    
    """
    print("Let's try hyperparameter tuning :")
    best_hyperparams, best_RMSE = RegressorHyperParameterSearch(data, output_label, k=5)
    print("Best hyper params  :  ", best_hyperparams)
    print("Best mean RMSE  :  ", best_RMSE)

    regressor = Regressor(x_train, nb_epoch = best_hyperparams[0], learning_rate=best_hyperparams[1], hidden_units=best_hyperparams[2], dropout_rate=best_hyperparams[3])
    losses = regressor.fit(x_train, y_train)

    save_regressor(regressor)

    # # Error
    print("Result on train : ")
    MSE, RMSE = regressor.score(x_train, y_train)
    print("MSE : ", MSE, "  |   RMSE : ", RMSE)
    print("Result on test : ")
    MSE, RMSE = regressor.score(x_test, y_test)
    print("MSE : ", MSE, "  |   RMSE : ", RMSE)
    """

if __name__ == "__main__":
    example_main()

