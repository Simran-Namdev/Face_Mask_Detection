# Face Mask Detection using TensorFlow and Keras
This project focuses on using machine learning and computer vision to detect the presence of face masks on individuals. The project includes the following key components:

## Libraries and Modules Used

- TensorFlow: TensorFlow is used for building and training deep learning models, specifically Convolutional Neural Networks (CNNs), for face mask detection.

- NumPy: NumPy is employed for numerical operations and data manipulation, particularly for working with image data.

- OpenCV: OpenCV is used for image processing tasks, such as reading and displaying images, and for preparing the dataset.

## Dataset Setup

- The project begins by importing necessary libraries and modules from TensorFlow, NumPy, and OpenCV.

- Directories for the dataset are set up, including directories for training, validation, and testing data. Each of these directories contains subdirectories for images with masks and without masks.

## Data Preprocessing

- ImageDataGenerator is utilized to create data generators for training, validation, and testing data.

- Data augmentation is applied to the training set, including techniques such as zooming, rotation, and horizontal flipping.

- All sets are normalized to ensure consistency and improve model training.

## Model Creation

- A Convolutional Neural Network (CNN) model is constructed using TensorFlow's Keras API.

- The model architecture consists of convolutional layers, max-pooling layers, dropout layers, and dense layers. This architecture is designed to maximize accuracy in mask detection.

## Model Compilation

- The model is compiled, specifying the optimizer (Adam), loss function (binary cross-entropy), and metrics (accuracy).

## Model Training

- The model is trained using the training data generator and validated using the validation data.

- Training continues for a specific number of epochs, with the model achieving an impressive accuracy of 99.62% on the validation set after 30 epochs.

## Model Evaluation

- The model's performance is evaluated on the test dataset, where it maintains a high accuracy of 99.19%.

- A confusion matrix and classification report provide additional insights into the model's performance.

## Inference and Predictions

- The trained model is used to make predictions on test images to determine the presence of masks.

- Sample images with and without masks are tested, demonstrating the model's ability to correctly predict mask presence.

