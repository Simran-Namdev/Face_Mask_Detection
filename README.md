# Face Mask Detection using TensorFlow and Keras

This project focuses on leveraging machine learning and computer vision techniques to detect the presence of face masks on individuals. The solution is built using TensorFlow and Keras, offering a robust framework for developing Convolutional Neural Networks (CNNs) tailored for mask detection.

## Key Components

### Libraries and Modules Utilized

- **TensorFlow:** The project harnesses the power of TensorFlow for constructing and training deep learning models, specifically CNNs, optimized for face mask detection tasks.

- **NumPy:** NumPy plays a crucial role in facilitating numerical operations and data manipulation, especially in handling image data effectively.

- **OpenCV:** OpenCV is instrumental in performing image processing tasks such as image reading, display, and dataset preparation, augmenting the dataset's robustness.

### Dataset Preparation

- The project initiates by importing essential libraries from TensorFlow, NumPy, and OpenCV to lay the groundwork for dataset preparation.

- Dataset directories are meticulously organized, comprising separate directories for training, validation, and testing data. Each directory contains subdirectories for images categorized into 'with masks' and 'without masks'.

### Data Preprocessing

- ImageDataGenerator is harnessed to generate data batches for training, validation, and testing.

- Data augmentation techniques, including zooming, rotation, and horizontal flipping, are applied to the training set to enhance model robustness.

- All datasets are normalized to ensure uniformity and optimize model training efficiency.

### Model Construction

- A sophisticated CNN model is constructed using TensorFlow's Keras API, integrating convolutional layers, max-pooling layers, dropout layers, and dense layers meticulously designed to maximize mask detection accuracy.

- Exploration of various pre-trained models such as ResNet50V2, ResNet152V2, InceptionV3, and Xception has been conducted to leverage their potential for improving model performance.

### Model Compilation and Training

- The model is compiled, specifying the Adam optimizer, binary cross-entropy loss function, and accuracy as the primary metric.

- Extensive model training is conducted using the training data generator, coupled with validation on a separate validation dataset.

- The model achieves an outstanding accuracy of 99.62% on the validation set after 30 epochs, showcasing its effectiveness in mask detection.

### Model Evaluation

- Rigorous evaluation is performed on the test dataset, demonstrating the model's robustness with a high accuracy of 99.19%.

- Additional insights into the model's performance are provided through a confusion matrix and a detailed classification report.

### Inference and Predictions

- The trained model is employed to make predictions on test images, accurately determining the presence of masks.

- A series of sample images are evaluated, demonstrating the model's proficiency in correctly predicting mask presence or absence.

## Additional Insights

### Transfer Learning Exploration

- The project delves into the realm of transfer learning, exploring various pre-trained models such as ResNet50V2, ResNet152V2, InceptionV3, and Xception.

- The weights of all transfer learning models are frozen, utilizing their feature extraction capabilities to enhance mask detection.

### Deployment

- The ResNet50V2 model, with a commendable accuracy of 99.3%, is deployed using Streamlit, offering users the flexibility to make predictions by uploading images or through live video streaming.

- For live video prediction, frame-by-frame prediction is implemented, ensuring real-time detection accuracy and reliability.

---
