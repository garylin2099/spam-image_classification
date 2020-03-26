# =============================================================================
# Preparation
# =============================================================================
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import io

np.random.seed(2020)

data = io.loadmat("data/data.mat")
X = data['X']
y = data['y'].ravel()

n = len(X)
# shuffle the data
shuffle_index = np.random.choice(n, n, replace=False)
X = X[shuffle_index]
y = y[shuffle_index]

X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
X = np.append(X, np.ones((n,1)), 1)

def cost_fun(X, y, w, Lambda):
#    a = np.log(expit(X @ w))
#    b = np.log(1 - expit(X @ w))
    w_prime = w[:(len(w)-1)]
    return(Lambda / 2 * w_prime.T @ w_prime - y.T @ np.log(expit(X @ w)) -
           (1 - y).T @ np.log(1 - expit(X @ w)))

def gradient(X, y, w, Lambda):
    return(Lambda * np.append(w[:(len(w)-1)], 0) - X.T @ (y - expit(X @ w)))

# =============================================================================
# Batch gradient descent
# =============================================================================
epsilon = 0.001
Lambda = 0.1
w = np.zeros(X.shape[1])

cost_batch = np.zeros(5000)
for i in range(5000):
    cost_batch[i] = cost_fun(X, y, w, Lambda)
    w = w - epsilon * gradient(X, y, w, Lambda)

plt.figure()
plt.plot(range(5000), cost_batch)
plt.title('Cost vs Number of Iteration (Batch Descent)')
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')


# =============================================================================
# stochastic gradient descent
# =============================================================================
epsilon = 0.01
np.random.seed(2020)
cost_stoch = np.zeros(5000)
w = np.zeros(X.shape[1])
for i in range(5000):
    random_index = np.random.choice(n)
    X_i = X[random_index].reshape(1,len(X[random_index]))
    y_i = np.array(y[random_index]).reshape(1)
    cost_stoch[i] = cost_fun(X, y, w, Lambda)
    w = w - epsilon * gradient(X_i, y_i, w, Lambda)
plt.figure()
plt.plot(range(5000), cost_stoch)
plt.title('Cost vs Number of Iteration (Stochastic Descent)')
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')


# =============================================================================
# Shrinking step size
# =============================================================================
np.random.seed(2020)
delta = 1
cost_stoch = np.zeros(5000)
w = np.zeros(X.shape[1])
for i in range(5000):
    epsilon = delta / (i+1)
    random_index = np.random.choice(n)
    X_i = X[random_index].reshape(1,len(X[random_index]))
    y_i = np.array(y[random_index]).reshape(1)
    cost_stoch[i] = cost_fun(X, y, w, Lambda)
    w = w - epsilon * gradient(X_i, y_i, w, Lambda)
plt.figure()
plt.plot(range(5000), cost_stoch)
plt.title('Cost vs # of Iteration (Stochastic Descent w/ Variable Step Size)')
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')



# =============================================================================
# Kaggle
# =============================================================================
# cross validation
lambda_vec = np.zeros(10)
j = -5
for i in range(10):
    lambda_vec[i] = 10 ** j
    j += 1

cv_accr_lambda = []
for j in lambda_vec:
    accr_cv = []
    for i in range(5):
        # partition training data into 5 sets
        val_start_ind = round(i/5*n)
        val_end_ind = round((i+1)/5*n)
        val_set_X = X[val_start_ind:val_end_ind]
        val_set_y = y[val_start_ind:val_end_ind]
        train_set_X = np.delete(X, range(val_start_ind,val_end_ind), axis=0)
        train_set_y = np.delete(y, range(val_start_ind,val_end_ind), axis=0)
        
        w = np.zeros(X.shape[1])
        for i in range(1000):
            w = w - epsilon * gradient(X, y, w, j)
        
        y_pred = val_set_X @ w
        y_pred = (y_pred >= 0.5) * 1
        accr_cv.append(metrics.accuracy_score(val_set_y, y_pred))
    # the average of the 5 cv errors given this lambda
    cv_accr_lambda.append(np.mean(accr_cv))

cv_accr_lambda_df = pd.DataFrame({'lambda': lambda_vec, 'accuracies': cv_accr_lambda})

# Prediction
epsilon = 0.001
Lambda = 0.1
w = np.zeros(X.shape[1])
# train the classifer
w = np.zeros(X.shape[1])
for i in range(1000):
    w = w - epsilon * gradient(X, y, w, Lambda)

# load test data, normalize, and add ficticious dimension
X_test = data['X_test']
X_test = (X_test - np.mean(X_test, axis = 0)) / np.std(X_test, axis = 0)
X_test = np.append(X_test, np.ones((len(X_test),1)), 1)

# make prediction
y_test_pred = X_test @ w
y_test_pred = (y_test_pred >= 0.5) * 1

def results_to_csv(y_test, filename):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv(filename, index_label='Id')

results_to_csv(y_test_pred, 'submission_batch01.csv')
    
    