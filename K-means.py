import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(12345)


# Function for loading the iris data
# load_data returns a 2D numpy array where each row is an example
#  and each column is a given feature.
def load_data():
    iris = datasets.load_iris()
    return iris.data

X = load_data()

def euclid_distance(p, q):
    return sum(list(map(lambda x: np.square(x[0] - x[1]), zip(p, q))))

# Assign labels to each example given the center of each cluster
def assign_labels(X, centers):
    return list(map(lambda x: min(list(zip(list(range(len(centers))), list(map(lambda y: euclid_distance(y, x), centers)))), key=lambda t: t[1])[0], X))

# Calculate the center of each cluster given the label of each example
def calculate_centers(X, labels):
    centroids = []
    clusters = list(set(labels))
    X = list(zip(X, labels))
    X = [list(elem) for elem in X]
    # print(X)
    for i in range(len(clusters)):
        # print(i)
        # print(X[i][1])
        # print(X)
        # for j in range(len(X)-1):
        # print(X[0][1])
        #get the all points of the current cluster
        current_cluster_points = list(filter(lambda x: x[1] == i, X))
        #get rid of the label
        current_cluster_points = list(map(lambda x: x[0], current_cluster_points))
        #calculte the value of the new centroid of this cluster by computing the mean
        current_centroid = sum(current_cluster_points)/len(current_cluster_points)
        centroids.append(current_centroid)

    return centroids


# Test if the algorithm has converged
# Should return a bool stating if the algorithm has converged or not.
def test_convergence(old_centers, new_centers):
    for i in range(len(new_centers)):
        if((old_centers[i] != new_centers[i]).any()):
            return False
    return True



# Evaluate the performance of the current clusters
# This function should return the total mean squared error of the given clusters
def evaluate_performance(X, labels, centers):
    X = list(zip(X, labels))
    X = [list(elem) for elem in X]
    result = 0
    for i in range(len(centers)):
        #get the all points of the current cluster
        current_cluster_points = list(filter(lambda x: x[1] == i, X))
        #get rid of the label
        current_cluster_points = list(map(lambda x: x[0], current_cluster_points))
        sum_of_squared_error_of_one_cluster = sum(list(map(lambda x: np.square(x - centers[i]), current_cluster_points)))
        result += sum_of_squared_error_of_one_cluster

    return sum(result)

# Algorithm for performing K-means clustering on the given dataset
def k_means(X, K):
    oldcenters = [1]
    newcenters = []
    labels = []
    #find random centroids
    centroid_indexs = random.sample(range(0, len(X)-1), K)
    for i in range(K):
        # c = X[random.randint(0, len(X)-1)]
        newcenters.append(X[centroid_indexs[i]])
    while not test_convergence(oldcenters, newcenters):
        oldcenters = newcenters
        #assign datapoints to clusters
        labels = assign_labels(X, newcenters)
        #calculate new centers
        newcenters = calculate_centers(X, labels)

    return (X, labels, newcenters)

#we test the kmeans algorithm with different amounts of clusters to find out with the elbow-method, which
#amount is most appropriated for our data.
#we calculate the mean-square-error for each K several times and compute an average error for this
#specific K, which we later plot
def test_k_means(X, number_of_clusters_to_test):
    listwithCurrentResult = [[], []]
    listwithResults = []
    for j in range(20):
        for i in range(number_of_clusters_to_test):
            listwithCurrentResult[0].append(i+1)
            k_means_result = k_means(X, i+1)
            listwithCurrentResult[1].append(evaluate_performance(X, k_means_result[1], k_means_result[2]))
        listwithResults.append(listwithCurrentResult)
        listwithCurrentResult = [[], []]
    #compute average of the errors
    listwithResults = [listwithResults[0][0], list(map(lambda x: x/len(listwithResults), [sum(m) for m in zip(*list(map(lambda x: x[1], listwithResults)))]))]
    return listwithResults

#Plotting the results of the elbow method
testresults = test_k_means(X, 10)
ax = plt.subplot(111)
ax.plot(testresults[0], testresults[1])
plt.title('K-Means Experiment')
ax.set_xlabel("number of clusters")
ax.set_ylabel("mean-square-error")
plt.show()





