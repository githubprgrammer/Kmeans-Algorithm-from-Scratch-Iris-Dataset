# Kmeans-Classifier-from-Scratch-Iris-Dataset
Implementing the kmeans algorithm to identify flower species from the Iris data set.

## Iris Dataset
The Iris flower data set is a multivariate data set which consists of 50 samples from each of three species of Iris. The variables measured for each sample are: the length and width of the sepals and petals, in centimeters. The combination of these four features is sufficient to develope a model for distinguishing the species from each other. The dataset was obtained from the scikit-learn library. 
It is accessible on the following website: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

## Implementation of Kmeans
The Kmeans algorithm consists of two main parts. 
- In the first step it assigns data points to clusters based on the current centroids. 
- Then it updates the centroids based on the current assignment of data points to clusters.

## How to choose an optimal K?
A general problem with the kmeans algorithm is selecting the K. The K represents the number of clusters we want the algorithm to produce. Since kmeans is a clustering algorithm and clustering is a problem of unsupervised learning we do not know beforehand what the correct answer/partitioning is. 
If we had domain knowledge we might have selected K manually. However without domain knowledge we have to use other methods to find a suitable K.

## The elbow method
One solution to this problem is the so called "elbow method" where  we have multiple clustering runs using different Ks. In our implementation we run our k-means-algorithm for several K's and then calculate the mean-square-error for each K. In order to get a more precise value for the mean-square-error for the different K’s , since the k-means-algorithm is dependent on the random initial centroids, we have run the algorithm for each K 20 times and then calculated the average mean-square-error over these 20 results.

### Result
In the figure below one can see the results: By applying the elbow-method we come to the conclusion that K=2 is a suitable amount of Clusters with a mean-square-error of approximately 150. After K=2 the error decreases only slowly. However another smaller “elbow” can be observed at K=4 with an error of approximately 60. After K=4 the error remains almost constant. Therefore depending on the accuracy needed we choose either 2 or 4 Clusters for the given dataset (assuming of course that we do not have any domain knowledge, e.g. knowing that there are only three species of iris existing and therefore K should be 3).

![alt text](https://github.com/githubprgrammer/Kmeans-Algorithm-from-Scratch-Iris-Dataset/blob/master/results%20of%20elbow%20method.png)
