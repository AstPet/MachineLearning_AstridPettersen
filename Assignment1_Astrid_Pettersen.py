''' ----- Assignment 1 - Music Genre Debate ----- '''
"           Author: Astrid Pettersen                "

#Importing necessary libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Some sklearn-function we need for easier testing, visualisation and cleanup, not for building the model itself
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay




"Problem 1"
'1a)'
#Importing the data using pandas to get a pandas dataframe containing all the data
file_with_path = ''
data_spotify = pd.read_csv(file_with_path, delimiter=',', encoding='utf-8')

#Finding the number of songs and the number of features using the built in shape function and printing
n_songs, n_features = data_spotify.shape[0], data_spotify.shape[1]
print(f'Number of songs in file: {n_songs}') 
print(f'Number of features in file: {n_features}')



'1b)'
#Defining new dataframes with only the songs with genres 'Pop' and 'Classical' in the first column (index 0)
data_genre = data_spotify[data_spotify.iloc[:, 0].isin(['Pop', 'Classical'])]

#Redifining the genres into 'Pop'= 1  and 'Classical' = 0
genre_mapping = {'Pop': 1, 'Classical': 0}
genre_numeric = np.array(data_genre['genre'].map(genre_mapping))

#Finding the number of songs in each of the two genres and printing
n_pop = np.sum(genre_numeric == 1)
n_classical = np.sum(genre_numeric == 0)
print(f'Number of pop songs in dataset: {n_pop}')
print(f'Number of classical songs in dataset: {n_classical}')

#Reducing the new dataset to only contain the features 'liveness' and 'loudness'
features_wanted = ['liveness', 'loudness']
data_reduced = data_genre[features_wanted]



'1c)'
#Defining our features as X and our genres as Y using arrays
X = np.array(data_reduced) #Features
Y = genre_numeric  #Songs' genre (is already an array)

#Splitting my data into 80/20 train/test and shuffling the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True, stratify=Y, random_state=42)



'1d) [Bonus]'
#Defining the classes seperately
Pop_data = data_genre[data_genre.iloc[:, 0] == 'Pop']
Classical_data = data_genre[data_genre.iloc[:, 0] == 'Classical']

#Plotting 'liveness vs. loudness'
def Plot_LivenessLoudness(Classical_data, Pop_data):
    plt.scatter(Classical_data.loc[:, 'liveness'], Classical_data.loc[:, 'loudness'], color='aqua', marker='x', label='Classical', s=5)
    plt.scatter(Pop_data.loc[:, 'liveness'], Pop_data.loc[:, 'loudness'], color='pink', marker='o', label='Pop', s=5)
    plt.xlabel('liveness')
    plt.ylabel('loudness')
    plt.title('Liveness vs. Loudness')
    plt.legend()
    plt.show()
Plot_LivenessLoudness(Classical_data, Pop_data)







"Problem 2"
'2a)'
class LogisticRegressionModel:
    def __init__(self, learning_rate=0.001, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs #Number of full iterations
        self.W = None #Weights
        self.B = None #Bias
        self.epsilon = 0.000001 #Added to avoid error in sigmoid function and log
    
    def SGD(self, X, Y):
        #Defining number of songs (rows) and features (columns) just for the simplicity of it
        songs = X.shape[0]
        features = X.shape[1]
        #Initialize weights and bias
        self.W = np.zeros(features)
        self.B = 0 #Bias

        #Defining an empty array for plotting cost as a function of epochs later
        error_array = np.zeros(self.epochs) 

        for epoch in range(self.epochs):
            for i in range(songs):
                #Calculationg our predicted Y
                Z = np.dot(self.W, X[i]) + self.B
                Y_hat = 1 / (1 + np.exp(-Z)) #Sigmoid function
                #Calculating error since we want to plot it
                error_array[epoch] -= (Y[i]*np.log(Y_hat+self.epsilon) + (1-Y[i])*np.log(1-Y_hat+self.epsilon))
                #Updating W and B by finding their gradients (dW and dB)
                dW = (Y_hat - Y[i]) * X[i]
                dB = Y_hat - Y[i]
                self.W -= self.lr*dW
                self.B -= self.lr*dB
        #Returning our trained values
        return self.W, self.B, error_array
    
    #Prediction of Y using our trained values for W and B
    def predict(self, X):
        Z = np.dot(self.W, X.T) + self.B
        Y_hat = 1 / (1 + np.exp(-Z))  # Sigmoid function
        Y_predict = np.where(Y_hat >= 0.5, 1, 0) #Threshold to classify as 0 or 1
        return Y_predict

    #Function for evaluation of 
    def evaluate(self, X, Y):
        Y_predicted = self.predict(X)
        accuracy = accuracy_score(Y, Y_predicted)
        return accuracy
        

#Calling our class
model = LogisticRegressionModel()
#Training our model on the training set
W_trained, B_trained, error_array = model.SGD(X_train, Y_train)

#Making a function for plotting our error
def PlotErrorEpoch(error_array):
    epochs_array = np.arange(len(error_array))
    plt.plot(epochs_array, error_array, color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error as a function of epochs')
    plt.show()
PlotErrorEpoch(error_array)

#Testing accuracy of our model on our training set
Training_accuracy = model.evaluate(X_train, Y_train)
print('The accuracy of our model on our training set is:', Training_accuracy)




'2b)'
#Testing my model on the test set
Testing_accuracy = model.evaluate(X_test, Y_test)
print('The accuracy of our model on our test set is:', Testing_accuracy)



'2c) [Bonus]'
#I struggled with this one




"Problem 3"
'3a)'
#Fetching the predicted Y for test set
Y_predicted = model.predict(X_test)
#Usink sklearn to create a confusion matrix
cm = confusion_matrix(Y_test, Y_predicted)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pop', 'Classical'])
display.plot(cmap='pink')
plt.title('Confusion Matrix')
plt.show()

