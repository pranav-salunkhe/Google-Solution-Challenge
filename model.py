import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load the data
data = pd.read_csv('csv_result-JM1.csv')
data = data.replace({'N': 0, 'Y': 1})

# Split the data
num_features = data.shape[1]
X, Y = data.iloc[:, 1:num_features-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=14)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy_nn = model.evaluate(X_test, y_test)[1]
print("Accuracy score for Neural Network:", accuracy_nn)

# GWO for feature selection with Neural Network
def fitness_function_nn(positions):
    features = np.where(positions >= 0.5)[0]
    train_xf = X_train[:, features]
    test_xf = X_test[:, features]

    # Define the neural network model for feature-selected data
    model_nn = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(train_xf.shape[1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile and train the model for feature-selected data
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_nn.fit(train_xf, y_train, epochs=7, batch_size=32)

    # Evaluate the model on the test set
    accuracy = model_nn.evaluate(test_xf, y_test)[1]
    return -accuracy  # Minimize the negative accuracy


import random
import numpy
import math
import time


def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):


    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30
    #SearchAgents_no=5

    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float("inf")

    Beta_pos=numpy.zeros(dim)
    Beta_score=float("inf")

    Delta_pos=numpy.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    #Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0,1,SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve=numpy.zeros(Max_iter)


     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")

    timerStart=time.time()
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])

            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()


            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()


            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score):
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()




        a=2-l*((2)/Max_iter); # a decreases linearly from 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):

                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]

                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)

                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1

                r1=random.random()
                r2=random.random()

                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)

                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2

                r1=random.random()
                r2=random.random()

                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)

                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3

                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)




        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
               print('alpha:', numpy.where(Alpha_pos>0.5)[0])

    timerEnd=time.time()
    print('Completed in', (timerEnd - timerStart))


    return Alpha_pos

# Use GWO to select features for the Neural Network
fit_nn = GWO(fitness_function_nn, 0, 1, X_train.shape[1], 10, 5)
selected_features_nn = np.where(fit_nn > 0.5)[0]
train_x1_nn = X_train[:, selected_features_nn]
test_x1_nn = X_test[:, selected_features_nn]

# Define a new neural network model for feature-selected data
model_selected = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(train_x1_nn.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model for feature-selected data
model_selected.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_selected.fit(train_x1_nn, y_train, epochs=7, batch_size=32)

# Evaluate the model on the test set
accuracy_nn_gwo = model_selected.evaluate(test_x1_nn, y_test)[1]
print("Modified Accuracy score for Neural Network with GWO:", accuracy_nn_gwo)


#Accuracy score for Neural Network: 0.7987909913063049
#Modified Accuracy score for Neural Network with GWO: 0.7996546030044556