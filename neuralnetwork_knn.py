import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer


# Load the datasets
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Explore the dataset
print("Number of unique classes:", train_data['label'].nunique())
print("Number of features:", len(train_data.columns) - 1)  # Excluding the label column
print("Missing values in training set:", train_data.isnull().sum().sum())

#X_test->test_images, y_test->test_labels

# Handle missing values by replacing NaN with mean
X_train = train_data.iloc[:, 1:].values
X_test  = test_data.values[:, 1:]

# Use SimpleImputer to replace NaN with mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Convert to float32 explicitly
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

# Normalize pixel values
X_train /= 255.0
X_test /= 255.0

# Reshape images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# Encode labels
y_train = to_categorical(train_data['label'])
y_test  = to_categorical(test_data['label'])

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# Visualize some images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {train_data['label'][i]}")
    ax.axis('off')
plt.show()

from sklearn.metrics import confusion_matrix
#Code steps
    #Builds a neural network model.
    #Compiles the model.
    #Trains the model on training data.
    #Evaluates the model on the test set.
    #Prints the test accuracy.
    #Calculates and prints the confusion matrix.

# Experiment 1

# Build the ANN model
model_1 = Sequential()
model_1.add(Flatten(input_shape=(28, 28, 1)))  # Flatten the 28x28 images
model_1.add(Dense(128, activation='relu'))
model_1.add(Dense(10, activation='softmax'))   # Output layer with 10 classes
#The output layer is added with 10 neurons (representing the 10 classes in the MNIST dataset)


# Compile the model
model_1.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
Train_model_1 = model_1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate on test set
test_loss_1, test_accurecy_1 = model_1.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accurecy_1}\n")

# Confusion Matrix
y_predict1 = model_1.predict(X_test)
Confusion_matrix_1 = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predict1, axis=1))
print("Confusion Matrix (Experiment 1):\n", Confusion_matrix_1)

#Hyperparameters:
    #Learning Rate: Determines the step size during optimization.
    #Batch Size: Number of training examples used in one iteration.
    #Number of Layers and Neurons: Architecture of the neural network.
    #Activation Functions: Choice of activation functions in each layer.
    #Dropout Rate: Regularization technique to prevent overfitting.
    #Weight Initialization: Initial values assigned to the weights.

# Experiment 2

# Build the ANN model
model_2 = Sequential()
model_2.add(Flatten(input_shape=(28, 28, 1)))
model_2.add(Dense(256, activation='relu'))  # Different number of neurons
model_2.add(Dense(10, activation='softmax'))

# Compile the model
model_2.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
Train_model_2 = model_2.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on test set
test_loss_2, test_accurecy_2 = model_2.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accurecy_2}\n")

# Confusion Matrix
y_predict2 = model_2.predict(X_test)
Confusion_matrix_2 = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predict2, axis=1))
print("Confusion Matrix (Experiment 2):\n", Confusion_matrix_2)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Flatten images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Convert one-hot encoded labels back to integers
y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Define the K-NN model
knn_model = KNeighborsClassifier()

# Define hyperparameter grid for grid search
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# Perform grid search
grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_flat, y_train_int)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
best_knn_model = grid_search.best_estimator_
best_knn_model.fit(X_train_flat, y_train_int)

# Predict on the validation set
knn_val_predictions = best_knn_model.predict(X_test_flat)

# Evaluate the model on the validation set
test_accuracy_3 = accuracy_score(y_test_int, knn_val_predictions)
print(f"Validation Accuracy (K-NN): {test_accuracy_3}")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test_int, knn_val_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_int, knn_val_predictions))

!pip install joblib

from tensorflow.keras.models import load_model

import joblib
from sklearn.metrics import confusion_matrix

# Compare the outcomes
if test_accurecy_1 > test_accurecy_2 and test_accurecy_1 > test_accuracy_3:
    BestModel = model_1
    print("Experiment 1 has the higher accuracy on the validation set. ANN model_1\n")
elif test_accurecy_2 > test_accurecy_1 and test_accurecy_2 > test_accuracy_3:
    BestModel = model_2
    print("Experiment 2 has the higher accuracy on the validation set. ANN model_2\n")
else:
    BestModel = knn_model
    print("Experiment 3 has the higher accuracy on the validation set. KNN model\n")

# Save the best model
if isinstance(BestModel, KNeighborsClassifier):
    joblib.dump(BestModel, "Best_model_knn.joblib")
else:
    BestModel.save("Best_model_Ann.h5")

# Reload the model
if isinstance(BestModel, KNeighborsClassifier):
    loaded_model = joblib.load("Best_model_knn.joblib")
else:
    loaded_model = load_model("Best_model_Ann.h5")

# Display the model summary
if not isinstance(BestModel, KNeighborsClassifier):
    loaded_model.summary()

# Evaluate the reloaded model on the test set
if isinstance(BestModel, KNeighborsClassifier):
    knn_val_predictions = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, knn_val_predictions)
    print(f"\nTest Accuracy using the reloaded KNN model: {accuracy}")
    # Get confusion matrix
    cm = confusion_matrix(y_test, knn_val_predictions)
    print("\nConfusion Matrix:")
    print(cm)
else:
    test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy using the reloaded ANN model: {test_accuracy}")
