import argparse
from os import listdir
from PIL import Image
import numpy as np
import csv


def setup_arg_parser():
    '''
    from template
    '''
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
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
    return args.train_path, args.test_path, args.o 

class NaiveBayes:
    '''
    Classifier Object used for task. Uses Principle Component Analysis (PCA) and Gaussian Naive Bayes
    '''


    def __init__(self, train_path, test_path, o, pca_var=0.85):
        '''
        Initializes internal variables.
        '''

        # Variables directly from function input
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = o
        self.pca_var = pca_var

        # Custom Variables to be initialized 
        self.truth_dict = dict()
        self.classes = list()
        self.raw_data = list()
        self.n_samples, self.n_features = int(), int()


        # Set up truth dict
        with open(self.train_path + '/' + "truth.dsv", 'r') as file:
            self.truth_dict = csv.reader(file, delimiter=':')
            self.truth_dict = dict(self.truth_dict)

        # Get the class labels
        self.classes = sorted(set(self.truth_dict.values()))


        # Initit 2D Array of raw data

        train_images_names = list(self.truth_dict.keys())

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
        Gets the mean, variation, and priors from the raw_data
        '''
        
        # Projects the Data after doing PCA
        projected_data = self._pca_transform(self.raw_data, self.pca_var)

        # Initialize the mean, var, and prior
        self.mean = np.zeros((len(self.classes), len(projected_data[0])), dtype=np.float64)
        self.var = np.zeros((len(self.classes), len(projected_data[0])), dtype=np.float64)
        self.priors = np.array([1/len(self.classes)]*len(self.classes))

        # Set the mean, var, and prior
        for idx, class_name in enumerate(self.classes):
            class_data = projected_data[np.array(list(self.truth_dict.values())) == class_name]
            self.mean[idx] = np.mean(class_data, axis=0)
            self.var[idx] = np.var(class_data, axis=0)


    def predict(self):
        '''
        Public Prediction function, also writes to a dsv. Wrapper for _predict
        '''

        # List of names of test images
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
        
        # Write to the output file
        with open(self.output_path, 'w') as file:
            for image_id, label in enumerate(ans_array):
                file.write(f"{test_images[image_id]}:{label}\n")


    def _predict(self, feature_list):
        """
        Calculate the probability for each class based on Gaussian Naive Bayes model.
        """

        # The following line is the single line version of
            # probabilities = []
            # for n_class in range(len(self.classes)):
            #     prior = np.log(self.priors[n_class])
            #     posterior = 0
            #     for n_feature in range(len(feature_list)):
            #         if self.var[n_class][n_feature] == 0: continue; 
            #         posterior += np.log(self._pdf(n_class, n_feature, feature_list[n_feature]))
            #     posterior = posterior + prior
            #     probabilities.append(posterior)

            # def _pdf(self, n_class, n_feature, feature):
            #     mean = self.mean[n_class][n_feature]
            #     var = self.var[n_class][n_feature]
            #     numerator = np.exp(-((feature - mean) ** 2) / (2 * var))
            #     denominator = np.sqrt(2 * np.pi * var)
            #     return numerator / denominator

        log_probs = np.log(self.priors) - 0.5 * np.sum(((feature_list - self.mean) ** 2 / self.var) + np.log(2 * np.pi * self.var), axis=1)

        return self.classes[np.argmax(log_probs)]
    

    def _pca_transform(self, data, var_cumsum=0.85):
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

        # Determine the number of components needed to reach the cumulative variance threshold specified by 'var_cumsum'.
        num_components = np.where(np.cumsum(eigenvalues) / np.sum(eigenvalues) >= var_cumsum)[0][0] + 1
            
        # Keep only the top 'num_components' eigenvectors and mean vector
        self.eigenvectors = eigenvectors[:, :num_components]
        self.mean_vector = mean_vector

        # Transform the original data into a new space defined by the top principal components
        return data.dot(self.eigenvectors)
        



if __name__ == "__main__":
    # Get Params, Initialize, Train, then Predict 

    train_path, test_path, o = main()
    classifier = NaiveBayes(train_path, test_path, o, 0.92)
    classifier.train()
    classifier.predict()
