import argparse
from os import listdir
from PIL import Image
import numpy as np
import csv
import random


def setup_arg_parser():
    '''
    from template, added k argument
    '''
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-k', type=int, 
                            help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def main():
    ''' 
    Return input params
    '''
    parser = setup_arg_parser()
    args = parser.parse_args()
    return args.train_path, args.test_path, args.o, args.k

class KNN:
    '''
    Classifier Object used for task. Uses k-Nearest Neighbour
    '''

    def __init__(self, train_path, test_path, o, k=0, pca_var=0.78):
        '''
        Initializes internal variables.
        '''
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = o
        self.k = k
        self.pca_var = pca_var

        if k == None or k == 0:
            self.k = 1

        # Variables to be initialized 
        self.truth_dict = dict()
        self.classes = list()
        self.raw_data = list()
        self.n_samples, self.n_features = int(), int()
        self.labels = np.array([])


        # Get raw data labels
        train_images_names = []
        with open(self.train_path + '/' + "truth.dsv", 'r') as file:
            reader = csv.reader(file, delimiter=':')
            for row in reader:
                train_images_names.append(row[0])
                self.labels = np.append(self.labels, row[1])

        # Classes are just the unique labels
        self.classes = sorted(set(self.labels))

        # The following line is the single line version o
            # self.raw_data = np.empty((num_images, len(sample_array)), dtype=np.float32)
            # for idx, image_name in enumerate(train_images_names):
                # image_path = self.train_path + '/' + image_name
                # image = Image.open(image_path)
                # self.raw_data[idx] = np.array(image).flatten() / 255
        self.raw_data = np.array([np.array(Image.open(self.train_path + '/' + image_name)).flatten() / 255 for image_name in train_images_names])

        # Getting number of samples and features in the raw data
        self.n_samples, self.n_features = self.raw_data.shape

    def train(self):
        '''
        Transforms the data to prepare for the nearest neighbout algorithm
        '''

        # Projects the Data after doing PCA
        self.projected_data = self._pca_transform(self.raw_data, self.pca_var)

    def predict(self):
        '''
        Public Prediction function, also writes to a dsv. Wrapper for _predict
        '''

        # Get name of all test images
        test_images = [filename for filename in listdir(self.test_path) if not filename.endswith(".dsv")]

        # The following line is the single line version of
            # for image in test_images:
            #---Load the image and preprocess it
            # img_vector = np.array(Image.open(self.test_path + '/' + image)).flatten() / 255
            # img_vector = img_vector - self.mean_vector
            #---Project the features of the test image
            # feature_array = img_vector.dot(self.eigenvectors)
            
            # supposed_ans = self._predict(feature_array)
        ans_array = [self._predict(((np.array(Image.open(self.test_path + '/' + image)).flatten() / 255 ) - self.mean_vector).dot(self.eigenvectors)) \
                    for image in test_images]

        # Write to file
        with open(self.output_path, 'w') as file:
            for image_id, label in enumerate(ans_array):
                file.write(f"{test_images[image_id]}:{label}\n")


    def _predict(self, feature_array):
        '''
        Calculates the distances of the given image to every input image and returns class of closest one
        '''

        # Creates array of distances
        distances = [self._euclid_distance(observation, feature_array) for observation in self.projected_data]

        # Gets the classes of the nearest self.k images
        nearest_classes = [self.labels[i] for i in np.argsort(distances)[:self.k]]
        
        # Voting which one is the strongest
        return  max(list(nearest_classes), key=nearest_classes.count)

    def _euclid_distance(self, array1, array2):
        '''
        Simple euclidean distance function
        '''
        return np.linalg.norm(array1 - array2)
    

    def _pca_transform(self, data, n_components=0.85):
        """
        Perform PCA on the data to reduce dimensionality while retaining 'var_cumsum' fraction of the variance.
        """

        # Calculate the mean feature vector and center the data 
        mean_vector = np.mean(data, axis=0)
        data -= mean_vector
        
        # Compute the covariance matrix
        cov_matrix = np.cov(data.T)
        
        # Perform eigendecomposition of the covariance matrix to extract eigenvalues and eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order (of the eigenvalues)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Distinguish between absolute number of features and cumulative sum 
        if (n_components < 1):
            num_components = np.where(np.cumsum(eigenvalues) / np.sum(eigenvalues) >= n_components)[0][0] + 1
        else:
            num_components = n_components
            
        # Keep only the top 'num_components' eigenvectors and mean vector
        self.mean_vector = mean_vector
        self.eigenvectors = eigenvectors[:, :num_components]

        return data.dot(self.eigenvectors)
        



if __name__ == "__main__":
    # Get Params, Initialize, Train, then Predict 

    train_path, test_path, o, k = main()
    classifier = KNN(train_path, test_path, o, k, 0.8435)
    classifier.train()
    classifier.predict()
