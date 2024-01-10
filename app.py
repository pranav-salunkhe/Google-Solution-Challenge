from flask import Flask, render_template, request
import google.generativeai as genai
import ast
import numpy as np
import joblib

GOOGLE_API_KEY = "AIzaSyDox1ipebTDKbAgqKsRXlPARIpT1kDnvek"
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)


model = genai.GenerativeModel('gemini-pro')
code_ = """
def unreliable_function(a, b, c, d, e, f, g, h, i, j, k, l, m):
    # High cyclomatic complexity with nested if statements
    if a > 0:
        if b < 10:
            if c % 2 == 0:
                for n in range(d):
                    if e == n:
                        while f > 0:
                            g -= 1
                            h *= 2
                        if i < 5:
                            j += 1
    else:
        # Complicated calculations and operations with additional parameters
        result = (a + b * c + k * l) / (d - e + m) + (f ** 2)
        return result

    # Unclear and poorly documented code
    if g == 0 or h == 0 or i == 0 or j == 0 or k == 0 or l == 0 or m == 0:
        print("Warning: Some variables are zero or undefined.")

    # Excessive parameters and unclear purpose
    return a + b + c + d + e + f + g + h + i + j + k + l + m
"""

# Define the features and corresponding formulas
features_formulas = {
    0: 'count(lines with only white spaces)',
    1: 'count(decision points)',
    2: 'count(pairs of functions calling each other)',
    3: 'count(lines with both code and comments)',
    4: 'count(lines with only comments)',
    5: 'count(conditions)',
    6: 'E - N + 2P, where E is the number of edges, N is the number of nodes, and P is the number of connected components',
    7: 'CYCLOMATIC_COMPLEXITY / NUMBER_OF_LINES',
    8: 'count(decision points)',
    9: 'DECISION_COUNT / NUMBER_OF_LINES',
    10: 'a metric based on design characteristics',
    11: 'DESIGN_COMPLEXITY / NUMBER_OF_LINES',
    12: 'count(edges in the control flow graph)',
    13: 'a metric based on essential characteristics',
    14: 'ESSENTIAL_COMPLEXITY / NUMBER_OF_LINES',
    15: 'count(lines with executable code)',
    16: 'count(parameters in function/method definitions)',
    17: 'n1 * log2(n1) + n2 * log2(n2), where n1 is the number of distinct operators, and n2 is the number of distinct operands',
    18: '(n1 / 2) * (N2 / n2), where n1 is the number of distinct operators, N2 is the total number of operators and operands, and n2 is the number of distinct operands',
    19: 'HALSTEAD_DIFFICULTY * HALSTEAD_VOLUME',
    20: 'HALSTEAD_EFFORT / 3000',
    21: 'N',
    22: '2 * (n2 / N2) * (N / n1), where n1 is the number of distinct operators, N2 is the total number of operators and operands, and n2 is the number of distinct operands',
    23: 'HALSTEAD_EFFORT / 18',
    24: 'N * log2(n1 + n2), where N is the total number of operators and operands, n1 is the number of distinct operators, and n2 is the number of distinct operands',
    25: 'a metric based on maintenance characteristics',
    26: 'count(modified conditions)',
    27: 'count(multiple conditions)',
    28: 'count(nodes in the control flow graph)',
    29: 'CYCLOMATIC_COMPLEXITY / NODE_COUNT',
    30: 'count(operands)',
    31: 'count(operators)',
    32: 'count(unique operands)',
    33: 'count(unique operators)',
    34: 'count(lines)',
    35: '(LOC_COMMENTS / NUMBER_OF_LINES) * 100',
    36: 'NUMBER_OF_LINES',
}

# Load the selected features
selected_features = np.load('selected_features.npy')

# Get the selected feature indices and their corresponding formulas
selected_feature_formulas = {index: features_formulas[index] for index in selected_features}
fea_size = len(selected_feature_formulas)
print(selected_feature_formulas)




@app.route("/")
def index():
    code = request.form.get("code")
    global prompt
    prompt = f"""
    Analyze the following Python code and calculate the following metrics using the specified formulae:

    {selected_feature_formulas}

    {code}
    """

    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    response = model.generate_content(prompt)

    prompt_ = f"""
    Make a numpy array for the all values calculated for below features and if any value is not applicable, replace it with 0 instead of nan. Do not omit any of below feature values. The numpy must have {fea_size} number of values.
    {selected_feature_formulas}

    Return just the numpy array as a string in the following format:
    "[value_1, value_2, value_3, ...]"
    {response.text}
    """

    array_obj = model.generate_content(prompt_)

    values = ast.literal_eval(array_obj.text)

    # Convert the list of values to a numpy array

    numpy_array = np.array(values)
    # numpy_array


    vals = values.strip('[]').split(', ')

    # Convert the list of strings to a numpy array
    np_array = np.array([float(value) for value in vals])

    print(np_array)

    nds = np.array(np_array)
    # nds
    nds = nds.reshape(1,-1)
    # nds
    # ds = nds[selected_features_svm_cgwo]

    # ds = ds.reshape(1,-1)
    # p = md.svcclassifier.predict(nds)
    loaded_model = joblib.load('svm_model_gwo.joblib')
    p = loaded_model.predict(nds)

    if p == 1 :
        prediction = "Reliable"
    else:
        prediction = "Unreliable"

    # Retrieve user input from the form
    # user_input = [float(request.form.get("code"))]



    # # Convert the 1D array to 2D
    # user_input_array = np.array(user_input).reshape(1, -1)

    # # Use the scaler fitted on the training data to standardize user input
    # user_input_standardized = std_scaler.transform(user_input_array)

    # # Use the selected features and GWO model for feature selection
    # selected_user_input = user_input_standardized[:, selected_features]

    # # Perform prediction
    # prediction = svcclassifier.predict(selected_user_input)

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
# # Your provided code here
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

# # Load the data
# data = pd.read_csv('csv_result-JM1.csv')
# data = data.replace({'N': 0, 'Y': 1})

# cols=[data.columns[1:len(data.columns)-1]]

# # Split the data
# num_features = data.shape[1]
# X, Y = data.iloc[:, 1:num_features-1], data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=14)

# # Standardize the data
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# train_data, test_data = train_test_split(data)
# train_x, test_x, train_y, test_y = train_data.iloc[:, 1:num_features-1], test_data.iloc[:, 1:num_features-1], train_data.iloc[:, -1], test_data.iloc[:, -1]
# std_scaler = StandardScaler()
# train_x = std_scaler.fit_transform(train_x.to_numpy())
# train_x = pd.DataFrame(train_x, columns=cols)

# test_x = std_scaler.transform(test_x.to_numpy())
# test_x = pd.DataFrame(test_x, columns=cols)


# # Support Vector Machine (SVM)
# svcclassifier = SVC(kernel = 'sigmoid', random_state =0)
# svcclassifier.fit(X_train, y_train)
# y_pred = svcclassifier.predict(X_test)
# P = accuracy_score(y_pred,y_test)
# print("Accuracy score for SVM:",P)

# # GWO for feature selection with SVM
# def fitness_function_svm(positions):
#     features = np.where(positions >= 0.5)[0]
#     train_xf = X_train[:, features]
#     test_xf = X_test[:, features]
#     svcclassifier = SVC(kernel='sigmoid', random_state=50)
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


# fit = GWO(fitness_function_svm, 0, 1,train_x.shape[1], 10, 1)
# selected_features = np.where(fit>0.5)[0]
# train_x1 = train_x.iloc[:, selected_features]
# cols=[train_x1.columns[0:len(train_x1.columns)-1]]
# print(cols)
# test_x1 = test_x.iloc[:, selected_features]
# svcclassifier = SVC(kernel = 'sigmoid', random_state = 0)
# svcclassifier.fit(train_x1, train_y)
# y_pred = svcclassifier.predict(test_x1)
# accuracy1=accuracy_score(y_pred,test_y)
# print("Modified Accuracy score for SVM:",accuracy1)
# print("Earlier Accuracy score was", P)

# features = ['LOC_BLANK', 'BRANCH_COUNT', 'CALL_PAIRS', 'LOC_CODE_AND_COMMENT',
#             'LOC_COMMENTS', 'CONDITION_COUNT', 'CYCLOMATIC_COMPLEXITY',
#             'CYCLOMATIC_DENSITY', 'DECISION_COUNT', 'DECISION_DENSITY',
#             'DESIGN_COMPLEXITY', 'DESIGN_DENSITY', 'EDGE_COUNT',
#             'ESSENTIAL_COMPLEXITY', 'ESSENTIAL_DENSITY', 'LOC_EXECUTABLE',
#             'PARAMETER_COUNT', 'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY',
#             'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH',
#             'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
#             'MAINTENANCE_SEVERITY', 'MODIFIED_CONDITION_COUNT',
#             'MULTIPLE_CONDITION_COUNT', 'NODE_COUNT',
#             'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUM_OPERANDS', 'NUM_OPERATORS',
#             'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS', 'NUMBER_OF_LINES',
#             'PERCENT_COMMENTS', 'LOC_TOTAL']

# @app.route("/")
# def index():
#     return render_template("index.html", features=features)

# @app.route("/predict", methods=["POST"])
# def predict():

#     # Retrieve user input from the form
#     user_input = [float(request.form.get(feature)) for feature in features]

#     # Convert the 1D array to 2D
#     user_input_array = np.array(user_input).reshape(1, -1)

#     # Use the scaler fitted on the training data to standardize user input
#     user_input_standardized = std_scaler.transform(user_input_array)

#     # Use the selected features and GWO model for feature selection
#     selected_user_input = user_input_standardized[:, selected_features]

#     # Perform prediction
#     prediction = svcclassifier.predict(selected_user_input)

#     return render_template("result.html", prediction=prediction[0])

# if __name__ == "__main__":
#     app.run(debug=True)
