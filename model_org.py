import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import numpy as np
import time
import joblib

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

# Support Vector Machine (SVM)
svcclassifier = SVC(kernel='sigmoid', random_state=0)
svcclassifier.fit(X_train, y_train)
y_pred_svm = svcclassifier.predict(X_test)
accuracy_svm = accuracy_score(y_pred_svm, y_test)
print("Accuracy score for SVM:", accuracy_svm)
joblib.dump(svcclassifier, 'svm_model.joblib')

# Chaotic Grey Wolf Optimization (CGWO)
def chaotic_map(x, a=0.6, b=3.9):
    return a * x * (1 - x) + b

def CGWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)

    print("CGWO is optimizing \"" + objf.__name__ + "\"")

    timerStart = time.time()

    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            fitness = objf(Positions[i, :])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score) and (fitness < Beta_score):
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score) and (fitness > Beta_score) and (fitness < Delta_score):
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        a = chaotic_map((2 * l) / Max_iter)  # Using logistic chaotic map to update parameter 'a'

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score

        if (l % 1 == 0):
            print(['At iteration ' + str(l) + ' the best fitness is ' + str(Alpha_score)])
            print('alpha:', np.where(Alpha_pos > 0.5)[0])

    timerEnd = time.time()
    print('Completed in', (timerEnd - timerStart))

    return Alpha_pos

# CGWO for feature selection with SVM
def fitness_function_svm(positions):
    features = np.where(positions >= 0.5)[0]
    train_xf = X_train[:, features]
    test_xf = X_test[:, features]
    svcclassifier = SVC(kernel='sigmoid', random_state=0)
    svcclassifier.fit(train_xf, y_train)
    y_pred = svcclassifier.predict(test_xf)
    accuracy = accuracy_score(y_pred, y_test)
    return -accuracy

fit_svm_cgwo = CGWO(fitness_function_svm, 0, 1, X_train.shape[1], 10, 20)
selected_features_svm_cgwo = np.where(fit_svm_cgwo > 0.5)[0]

np.save('selected_features.npy', selected_features_svm_cgwo)


train_x1_svm_cgwo = X_train[:, selected_features_svm_cgwo]
test_x1_svm_cgwo = X_test[:, selected_features_svm_cgwo]
svcclassifier.fit(train_x1_svm_cgwo, y_train)
y_pred_svm_cgwo = svcclassifier.predict(test_x1_svm_cgwo)
accuracy_svm_cgwo = accuracy_score(y_pred_svm_cgwo, y_test)
print("Modified Accuracy score for SVM with CGWO:", accuracy_svm_cgwo)

joblib.dump(svcclassifier, 'svm_model_gwo.joblib')




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# import joblib 

# # Load the data
# data = pd.read_csv('csv_result-JM1.csv')
# data = data.replace({'N': 0, 'Y': 1})

# # Split the data
# num_features = data.shape[1]
# X, Y = data.iloc[:, 1:num_features-1], data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=14)

# # Standardize the data
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Support Vector Machine (SVM)
# svcclassifier = SVC(kernel='sigmoid', random_state=0)
# svcclassifier.fit(X_train, y_train)
# y_pred_svm = svcclassifier.predict(X_test)
# accuracy_svm = accuracy_score(y_pred_svm, y_test)
# print("Accuracy score for SVM:", accuracy_svm)


# # GWO for feature selection with SVM
# def fitness_function_svm(positions):
#     features = np.where(positions >= 0.5)[0]
#     train_xf = X_train[:, features]
#     test_xf = X_test[:, features]
#     svcclassifier = SVC(kernel='sigmoid', random_state=0)
#     svcclassifier.fit(train_xf, y_train)
#     y_pred = svcclassifier.predict(test_xf)
#     accuracy = accuracy_score(y_pred, y_test)
#     return -accuracy

# import random
# import numpy
# import math
# import time


# def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):


#     #Max_iter=1000
#     #lb=-100
#     #ub=100
#     #dim=30
#     #SearchAgents_no=5

#     # initialize alpha, beta, and delta_pos
#     Alpha_pos=numpy.zeros(dim)
#     Alpha_score=float("inf")

#     Beta_pos=numpy.zeros(dim)
#     Beta_score=float("inf")

#     Delta_pos=numpy.zeros(dim)
#     Delta_score=float("inf")

#     if not isinstance(lb, list):
#         lb = [lb] * dim
#     if not isinstance(ub, list):
#         ub = [ub] * dim

#     #Initialize the positions of search agents
#     Positions = numpy.zeros((SearchAgents_no, dim))
#     for i in range(dim):
#         Positions[:, i] = numpy.random.uniform(0,1,SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

#     Convergence_curve=numpy.zeros(Max_iter)


#      # Loop counter
#     print("GWO is optimizing  \""+objf.__name__+"\"")

#     timerStart=time.time()
#     # Main loop
#     for l in range(0,Max_iter):
#         for i in range(0,SearchAgents_no):

#             # Return back the search agents that go beyond the boundaries of the search space
#             for j in range(dim):
#                 Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])

#             # Calculate objective function for each search agent
#             fitness=objf(Positions[i,:])

#             # Update Alpha, Beta, and Delta
#             if fitness<Alpha_score :
#                 Alpha_score=fitness; # Update alpha
#                 Alpha_pos=Positions[i,:].copy()


#             if (fitness>Alpha_score and fitness<Beta_score ):
#                 Beta_score=fitness  # Update beta
#                 Beta_pos=Positions[i,:].copy()


#             if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score):
#                 Delta_score=fitness # Update delta
#                 Delta_pos=Positions[i,:].copy()




#         a=2-l*((2)/Max_iter); # a decreases linearly from 2 to 0

#         # Update the Position of search agents including omegas
#         for i in range(0,SearchAgents_no):
#             for j in range (0,dim):

#                 r1=random.random() # r1 is a random number in [0,1]
#                 r2=random.random() # r2 is a random number in [0,1]

#                 A1=2*a*r1-a; # Equation (3.3)
#                 C1=2*r2; # Equation (3.4)

#                 D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
#                 X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1

#                 r1=random.random()
#                 r2=random.random()

#                 A2=2*a*r1-a; # Equation (3.3)
#                 C2=2*r2; # Equation (3.4)

#                 D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
#                 X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2

#                 r1=random.random()
#                 r2=random.random()

#                 A3=2*a*r1-a; # Equation (3.3)
#                 C3=2*r2; # Equation (3.4)

#                 D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
#                 X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3

#                 Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)




#         Convergence_curve[l]=Alpha_score;

#         if (l%1==0):
#                print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
#                print('alpha:', numpy.where(Alpha_pos>0.5)[0])

#     timerEnd=time.time()
#     print('Completed in', (timerEnd - timerStart))


#     return Alpha_pos


# fit_svm = GWO(fitness_function_svm, 0, 1, X_train.shape[1], 10, 20)
# selected_features_svm = np.where(fit_svm > 0.5)[0]
# train_x1_svm = X_train[:, selected_features_svm]
# test_x1_svm = X_test[:, selected_features_svm]
# svcclassifier.fit(train_x1_svm, y_train)
# y_pred_svm_gwo = svcclassifier.predict(test_x1_svm)
# accuracy_svm_gwo = accuracy_score(y_pred_svm_gwo, y_test)
# print("Modified Accuracy score for SVM with GWO:", accuracy_svm_gwo)
