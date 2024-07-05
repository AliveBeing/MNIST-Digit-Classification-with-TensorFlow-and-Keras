# MNIST-Digit-Classification-with-TensorFlow-and-Keras
### MNIST Digit Classification with TensorFlow and Keras

This repository contains code to build and train a neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

#### Project Description
This project demonstrates a basic implementation of a neural network for digit classification using the MNIST dataset. The MNIST dataset is a standard benchmark for image classification tasks, consisting of 70,000 images of handwritten digits (0-9) split into training and test sets.

#### Code Overview
1. **Importing Libraries**: The necessary libraries such as TensorFlow, Keras, Matplotlib, and Numpy are imported. 
2. **Loading Data**: The MNIST dataset is loaded using Keras's built-in dataset loader.
3. **Data Preprocessing**: The pixel values of the images are normalized by dividing by 255 to scale them between 0 and 1. The training and test images are flattened from 28x28 to a 784-dimensional vector.
4. **Model Creation and Compilation**:
   - **First Model**: A simple neural network with a single Dense layer of 10 neurons and sigmoid activation.
   - **Second Model**: A more complex neural network with one hidden Dense layer of 100 neurons using ReLU activation and an output Dense layer of 10 neurons using sigmoid activation.
5. **Training the Models**: Both models are trained on the training data for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
6. **Evaluation and Prediction**: The second model is used to make predictions on the test data. The predictions are compared with the true labels to generate a confusion matrix.
7. **Visualization**: The confusion matrix is visualized using Seaborn's heatmap function to show the performance of the model.

#### How to Run
1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```sh
   cd mnist-digit-classification
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the script:
   ```sh
   python mnist_classification.py
   ```

#### Requirements
- TensorFlow
- Keras
- Matplotlib
- Numpy
- Seaborn

#### Results
The code includes a confusion matrix visualization to display the classification accuracy of the model on the test dataset.

#### Future Work
Potential improvements and extensions of this project include:
- Experimenting with different network architectures and hyperparameters.
- Implementing data augmentation techniques to improve model performance.
- Exploring advanced models like Convolutional Neural Networks (CNNs) for better accuracy.

Feel free to contribute to this project by submitting issues or pull requests!

