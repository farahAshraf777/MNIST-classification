The MNIST dataset is a widely used collection of handwritten digits, serving as a benchmark for various machine learning tasks. This project aims to leverage the Tensorflow framework and the Scikit-Learn library to explore and implement both K-Nearest Neighbors (K-NN) and Artificial Neural Network (ANN) algorithms for digit classification.

Approach:

Data Exploration and Preparation:

Load Dataset: The project begins by loading the MNIST dataset in CSV format, containing both training and test sets.

Initial Data Exploration:

Identify the number of unique classes (digits 0 to 9).
Identify the number of features (pixels).
Check for missing values.
Data Preprocessing:

Normalize pixel values by dividing each by 255.
Resize images to 28x28 dimensions.
Visualize resized images to ensure correctness.
Split the training data into training and validation sets.
Experiments and Results:

Experiment 1: K-Nearest Neighbors (K-NN)

Implement K-NN algorithm for digit classification.
Utilize grid search for hyperparameter tuning.
Evaluate and compare results.


Experiment 2: Artificial Neural Network (ANN)

Construct and train two ANN architectures with varying parameters:
Hidden neurons
Learning rate
Batch size
Compare outcomes with K-NN, identifying the best-performing model.
Outcome Analysis:

Compare the accuracies of K-NN and ANN on the validation dataset.
Generate the confusion matrix of the best model.
Model Saving and Testing:

Save the best-performing model.
Reload the saved model.
Evaluate the model on the testing data from mnist_test.csv.
Conclusion:
The project concludes with insights into the comparative performance of K-NN and ANN on the MNIST dataset, emphasizing the importance of hyperparameter tuning in achieving optimal results. The saved model can be used for future predictions on new data.

🔗 Download Dataset:
[MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
