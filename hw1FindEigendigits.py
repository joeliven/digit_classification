import scipy as sc
import scipy.io as scio
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import hw1FindEigendigits_functions as fn
import scipy.linalg as lin


# put data into separate lists:
raw_data = scio.loadmat("digits.mat")
trainImages = np.array(raw_data["trainImages"], dtype=np.float64)
trainLabels = np.array(raw_data["trainLabels"], dtype=np.float64)
testImages = np.array(raw_data["testImages"], dtype=np.float64)
testLabels = np.array(raw_data["testLabels"], dtype=np.float64)

# User Input Arguments, specifying details of experiment run:
# experiement ID:
hw1_exp_id = raw_input("experiment ID: ")
# how many training samples to use when generating the eigenspace mapping?
mapping_size = int(raw_input("use top X eigenvectors for mapping_size: (integer) "))
# include training data used in mapping in training set?
reuse_mapping_data = raw_input("reuse mapping data? (y/n) ")
# how many training samples to use for classification?
num_train_samples = int(raw_input("num_train_samples: (integer) "))
# how many test images to use for classification?
num_test_samples = int(raw_input("num_test_samples: (integer) "))
# use easy or difficult test data? or mixed?
test_data_dif_level = raw_input("test data difficulty level: (easy/hard) ")
# how many neighbors to use in k-nearest neighbor classification?
k_nn = int(raw_input("# of nearest neighbors: (integer) "))
# add header to results file?
add_header = raw_input("add header: (y/n) ")
# show images
show_mapping_images = int(raw_input("show every Xth mapping_image: (integer) "))
show_test_images = int(raw_input("show every Xth original test image: (integer) "))
show_reconstructed_test_images = int(raw_input("show every Xth reconstructed test image: (integer) "))

# prep the data by reshaping it to the appropriate dimensions and separating out the difficult and easy test data:
trainImages, testImages_difficult, testImages_easy, testLabels_difficult, testLabels_easy \
     = fn.prep_data(trainImages, testImages, testLabels)
    
A = trainImages[:, 0:mapping_size]

# # show training samples in A:
# for j in xrange(0, len(A[0, :])):
#     fn.show_digit(A[:, j])

# get a 784 x mapping_size matrix V that contains the top mapping_size eigenvectors of the covariance matrix of A:
m, V = fn.hw1FindEigenDigits(A)

# show eigendigits from training data used to create the mapping:
if show_mapping_images > 0:
    for j in xrange(0, len(V[0, :])):
        if (j % show_mapping_images == 0):
            fn.show_digit(V[:, j])

# Train a set of training data, by projecting the data points into the reduced eigenspace:
if reuse_mapping_data == 'y':
   trainingData = trainImages[:, 0:num_train_samples]
   trainingLabels = trainLabels[0, 0:num_train_samples] 
else:
    trainingData = trainImages[:, mapping_size:(mapping_size + num_train_samples)]
    trainingLabels = trainLabels[0, mapping_size:(mapping_size + num_train_samples)]

trainingData_eigspace = fn.process_trainingData(trainingData, trainingLabels, V, m) # returns a list of dictionaries

# grab the indicated number of test images to test on:
if test_data_dif_level == 'easy':
    test_data = testImages_easy[:, 0:num_test_samples]
    test_labels = testLabels_easy[0, 0:num_test_samples]
else:
    test_data = testImages_difficult[:, 0:num_test_samples]
    test_labels = testLabels_difficult[0, 0:num_test_samples]

test_labels = np.reshape(test_labels, (1,len(test_labels)))

# for each test vector:
    # project it into the reduced eigenspace (of dim = mapping_size x 1)
    # then classify it in that space by comparing to the training data projected into that space
    # then project back from eigenspace into full dim space = 784 x 1
    # then display the reconstruction (if option enabled by user)

# show testImages before being projected into and out of eigenspace (if option enabled by user):
if show_test_images > 0:
    for j in xrange(0, len(test_data[0, :])):
        if (j % show_test_images == 0):
            fn.show_digit(test_data[:, j])

# create file for output of individual experiment results:
flname = "hw1FindEigendigits_exp_" + hw1_exp_id + ".txt"
fl = open(flname, 'a')
fl.write(str(flname) + "\n")
incorrect_list = list()

correct = 0
incorrect = 0
for j in xrange(0, len(test_data[0, :])):
    test_vec = test_data[:,j] # grab the test vector from the test_data matrix
    test_vec_eigspace = fn.project_to_eigspace(test_vec, V, m) # project the vector into eigspace
    nearest_neighbors_sorted = fn.find_knn(test_vec_eigspace, trainingData_eigspace) # returns a list of dictionaries
    projected_target, votes = fn.classify(nearest_neighbors_sorted, k_nn) # get the projected_target based on k_nn voting
    actual_target = test_labels[0,j]
    # prep output to screen and file:
    msg1 = "votes are "
    print("\n" + str(j) + ")\n"+ str(msg1))
    print(votes)
    msg2 = "projected_target is: " + str(projected_target)
    msg3 = "actual_target is: " + str(actual_target)
    print(msg2)
    print(msg3)
    #output to file:
    fl.write("\n" + str(j) + ")\n" + str(msg1))
    fl.write(str(votes) + "\n")
    fl.write(str(msg2) + "\n")
    fl.write(str(msg3) + "\n")
    
    if projected_target == actual_target:
        correct += 1
        print("!!!!!!!!!!!!!!!CORRECT!!!!!!!!!!!!!!!")
        fl.write("!!!!!!!!!!!!!!!CORRECT!!!!!!!!!!!!!!!")
    else:
        incorrect += 1
        print("---------------incorrect---------------")
        fl.write("---------------incorrect---------------")
        incorrect_list.append(actual_target)
    # reconstruct the test vector back from eigspace to full space:
    test_vec_reconstructed = fn.project_from_eigspace(test_vec_eigspace, V)
    if show_reconstructed_test_images > 0:
        if (j % show_reconstructed_test_images == 0):
            fn.show_digit(test_vec_reconstructed)

# output results to stdout and file:
total = float(correct+incorrect)
percent_correct = correct / total
msg4 = "\n\n****************\n****************\npercent correct is " + str(correct) + " / " + str(correct+incorrect) + " = " + str(percent_correct)
print(msg4)
fl.write(msg4)
msg5 = "\n\ninncorrect_list=" + str(incorrect_list)
fl.write(msg5)
fl.close()

# output summar of results to master results file:
f = open("hw1_results.csv", 'a')
results = str(percent_correct) + "," \
    + str(correct) + "," \
    + str(total) + "," \
    + str(mapping_size) + "," \
    + str(reuse_mapping_data) + "," \
    + str(num_train_samples) + "," \
    + str(num_test_samples) + "," \
    + str(test_data_dif_level) + "," \
    + str(k_nn) + "," \
    + str(hw1_exp_id) + "\n"

if add_header == 'y':
    header_string = "percent_correct,correct,total,mapping_size,reuse_mapping_data,num_train_samples,num_test_samples,test_data_dif_level,k_nn\n" 
    f.write(header_string)
f.write(results)
f.close()