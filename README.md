# OCR Using Manually Implemented KNN and Naive Bayes

This repository contains Python implementations of the K-Nearest Neighbors (KNN) and Naive Bayes algorithms designed to perform Optical Character Recognition (OCR) on grayscale images. The program takes image data and a truth table in DSV format to train and then predict character outputs.

## Features

- **KNN Algorithm:** Customizable through the `-k` parameter to define the number of neighbors.
- **Naive Bayes Algorithm:** Implementation tailored for grayscale image data.
- **PCA:** Both Algorithms use Principle Component Analysis as pre-processing for the image data.
- **Flexible Data Input:** Specify training and testing paths directly through the command line.

## Prerequisites

Before running the scripts, ensure that you have Python installed on your system. You might also need additional libraries depending on the final implementation specifics. 
The training and test data need to be grayscale images of the same sizes.
