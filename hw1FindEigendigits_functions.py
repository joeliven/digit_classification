import scipy as sc
import scipy.io as scio
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import scipy.linalg as lin
import scipy.stats as st

def show_digit(col):
    col = np.reshape(col, (28, 28))
    im = plt.imshow(col)
    plt.show()
    
def prep_data(trainImages, testImages, testLabels):
    # reshape training digits from (28, 28, 1, 60000) to (784, 60000)
    trainImages = np.reshape(trainImages, (784, 60000))
    # reshape test digits from (28, 28, 1, 60000) to (784, 60000)
    testImages = np.reshape(testImages, (784, 10000))

    # separate out the "easy" and "difficult" test data sets:
    testImages_difficult = np.array(testImages[:, 0:5000])
    testImages_easy = np.array(testImages[:, 5000:10000])
    testLabels_difficult = np.array(testLabels[:, 0:5000])
    testLabels_easy = np.array(testLabels[:, 5000:10000])
    
    return trainImages, testImages_difficult, testImages_easy, testLabels_difficult, testLabels_easy

def print_shapes(trainImages, trainLabels, testImages_difficult, testImages_easy, testLabels_difficult, testLabels_easy):
    print("testing this setup")
    print("trainImages shape = " + str(np.shape(trainImages)))
    print("trainLabels shape = " + str(np.shape(trainLabels)))
    print("testImages_difficult shape = " + str(np.shape(testImages_difficult)))
    print("testImages_easy shape = " + str(np.shape(testImages_easy)))
    print("testLabels_difficult shape = " + str(np.shape(testLabels_difficult)))
    print("testLabels_easy shape = " + str(np.shape(testLabels_easy)))

def find_mean_col_vec(A):
    m = np.mean(A, axis=1, dtype=np.float64)
    m = np.reshape(m, (len(m),1)) 
    return m

def hw1FindEigenDigits(A):
    debug = False
    m = find_mean_col_vec(A)
    A = np.subtract(A, m) # subtract mean col vec, m, from each col of A
    A_t = np.transpose(A) # get the transpose of A
    A_t_A = np.dot(A_t, A) # get the product of A_t and A
    eig_vals, eig_vecs = lin.eig(A_t_A) # find the eigenvectors and eigenvalues of A_t_A
    if debug:
        print("eig_vals shape = " + str(np.shape(eig_vals)))
        print(eig_vals)
        print("eig_vecs shape = " + str(np.shape(eig_vecs)))
        print(eig_vecs)
    # Reference Note: the "idx" sorting trick used in the next 3 lines was adapted from an online source found on Overstack.com
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    if debug:
        print("sorted_eig_vals = ")
        print(eig_vals)
        print("sorted_eig_vecs = ")
        print(eig_vecs)
        print("eig_vals_shape")
        print(str(np.shape(eig_vals)))
        print("eig_vecs_shape")
        print(str(np.shape(eig_vecs)))
        test_V(eig_vecs)
    V = np.empty_like(A)
    V = np.mat(V)
    A = np.mat(A)
    for j in xrange(0, len(eig_vals)):
        vec = eig_vecs[:,j]
        vec = np.mat(np.reshape(vec, (len(vec),1)))
        V[:,j] = (A*vec) # matrix-vector multiplication in this case
    V = np.asarray(V)
    V = normalize_matrix(V)    
    return m, V

def normalize_matrix(V):
    for j in xrange(0, len(V[0,:])):
        norm = lin.norm(V[:,j])
        V[:,j] *= (1.0/norm)
    return V

def test_V(V):
    for j in xrange(0, len(V[0,:])):
        norm = lin.norm(V[:,j])
        print("norm of V[:, " + str(j) + "] = " + str(norm))
    
def process_trainingData(trainingData, trainingLabels, V, m):
    debug = False
    trainingLabels = np.reshape(trainingLabels, (1,len(trainingLabels)))
    if debug:
        print("shape of trainingLabels in process_trainingData is " + str(np.shape(trainingLabels)))
    trainingData_eigspace = list()
    if debug:
        print("len(trainingLabels) is " + str(len(trainingLabels[0,:])))
    for j in xrange(0, len(trainingLabels[0,:])):
        vec_eigspace = project_to_eigspace(trainingData[:,j], V, m)
        trainingData_eigspace.append(dict({'target': trainingLabels[0,j], 'vec_eigspace': vec_eigspace}))
    return trainingData_eigspace
       
def project_to_eigspace(vector, V, m):
    debug = False
    if debug:
        print("shape of vector is " + str(np.shape(vector)))
    vec = np.reshape(vector, (len(vector),1))
    if debug:
        print("shape of vec is " + str(np.shape(vec)))
        print("shape of m is " + str(np.shape(m)))
    vec = np.subtract(vec, m)
    if debug:
        print("shape of vec is " + str(np.shape(vec)))
    V_t = np.transpose(V)
    if debug:
        print("shape of V_t is " + str(np.shape(V_t)))
    vec_eigspace = np.dot(V_t, vec) # this is now a kx1 vector in the reduced eigenspace
    if debug:
        print("shape of vec_eigspace is " + str(np.shape(vec_eigspace)))
    return vec_eigspace

def project_from_eigspace(vec_eigspace, V):
    vec = np.reshape(vec_eigspace, (len(vec_eigspace),1))
    vec = np.dot(V, vec_eigspace)
    return vec
    
def find_knn(test_vec_eigspace, trainingData_eigspace):
    debug = False
    nearest_neighbors = list()
    if debug:
        print("shape of test_vec_eigspace is " + str(np.shape(test_vec_eigspace)))
        test_vec_eigspace = np.reshape(test_vec_eigspace, (len(test_vec_eigspace),1))
        print("shape of test_vec_eigspace is NOW " + str(np.shape(test_vec_eigspace)))
        print("trainingData_eigspace is...")
        print(trainingData_eigspace)
    for dicti in trainingData_eigspace:
        train_vec_eigspace = dicti['vec_eigspace']
        train_vec_eigspace = np.reshape(train_vec_eigspace, (len(train_vec_eigspace),1))
        difference = test_vec_eigspace - train_vec_eigspace
        distance = lin.norm(difference)
        nearest_neighbors.append(dict({'predicted_target': dicti['target'], 'distance': distance}))
    nearest_neighbors_sorted = sorted(nearest_neighbors, key=lambda entry: entry['distance'])
    return nearest_neighbors_sorted
     
def classify(nearest_neighbors_sorted, knn):
    top_nearest_neighbors = nearest_neighbors_sorted[0:knn]
    votes = list()
    for dicti in top_nearest_neighbors:
        votes.append(dicti['predicted_target'])

    winner, count = st.stats.mode(votes, axis=None)
    return winner, votes  