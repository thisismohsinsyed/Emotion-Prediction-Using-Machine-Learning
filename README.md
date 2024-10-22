
# Emotion Prediction using CNN

## Project Overview
This project focuses on predicting human emotions from facial images using a **Convolutional Neural Network (CNN)** model. The aim is to classify images into different emotion categories such as happy, sad, angry, and others. This repository contains the code and resources used to develop, train, and evaluate the CNN model for emotion recognition. The dataset used for this project was obtained from **Kaggle's Facial Emotion Recognition Dataset**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project is the **[Facial Expression Recognition (FER) dataset](https://www.kaggle.com/datasets/msambare/fer2013)**, which consists of grayscale images of human faces. The dataset is structured into several categories of emotions, including:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Each image is a 48x48 pixel grayscale image, and the dataset is labeled with the corresponding emotion.

### Dataset Details:
- Number of samples: 35,887
- Image resolution: 48x48 pixels (grayscale)
- Classes: 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

## Model Architecture
We used a **Convolutional Neural Network (CNN)** to perform the emotion classification. CNNs are highly effective in tasks involving image recognition because they can automatically capture spatial hierarchies and patterns in images.

The architecture used for this project includes:
- Input Layer: Grayscale image of size 48x48 pixels
- Convolutional Layers: Multiple layers with filters to detect spatial patterns
- Activation Function: ReLU (Rectified Linear Unit) after each convolution layer
- Pooling Layers: MaxPooling for downsampling
- Fully Connected Layers: Dense layers to perform classification
- Output Layer: Softmax activation to output probabilities for each emotion category

### Model Summary:
```
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D                        (None, 48, 48, 64)         640
MaxPooling2D                  (None, 24, 24, 64)         0
Conv2D                        (None, 24, 24, 128)        73856
MaxPooling2D                  (None, 12, 12, 128)        0
Flatten                       (None, 18432)              0
Dense                         (None, 512)                9437696
Dropout                       (None, 512)                0
Dense                         (None, 7)                  3591
=================================================================
Total params: 9,514,783
Trainable params: 9,514,783
Non-trainable params: 0
```

## Installation
To run this project, you'll need Python and the required libraries. Follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Emotion-Prediction-Using-Machine-Learning.git
cd Emotion-Prediction-Using-Machine-Learning

    ```
2. **Install dependencies**:
   Create a virtual environment (optional) and install the required packages using the `requirements.txt` file.
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   You can download the dataset from the following link: [Kaggle FER Dataset](https://www.kaggle.com/datasets/msambare/fer2013). Extract the dataset and place it in the `data/` directory.

## Usage
To train the CNN model on the emotion dataset and evaluate its performance, follow these steps:

1. **Preprocessing**: The preprocessing script loads the dataset, normalizes the images, and prepares them for training and testing.

2. **Training the Model**:
   To train the model, run the following:
  
  Emotion Detection using Machine Learning  (Training).ipynb

  For Testing
   

   The training process will save the best model weights to a file in the same directory.

3. **Evaluating the Model**:
   After training, you can evaluate the model on the test set:
   Emotion Prediction Using Machine Learning (Testing).ipynb

   This will output the accuracy, confusion matrix, and other performance metrics.

4. **Predicting Emotions**:
   You can use the trained model to predict emotions from new images:
  Emotion Prediction Using Machine Learning (Testing).ipynb 

## Results
The model was trained for 100 epochs, and it achieved the accuracy of 90%

### Sample Predictions:
Here are some examples of the model's predictions:

| Image | Predicted Emotion | Confidence |
|-------|-------------------|------------|
| img1.jpg | Happy | 90% |
| img2.jpg | Sad | 87% |
| img3.jpg | Angry | 90% |

The model generally performs well, especially in recognizing distinct emotions like happiness and anger, but struggles with subtler emotions like disgust and fear.

## Contributing
Contributions are welcome! If you have suggestions for improving the model or code, feel free to fork the repository and open a pull request. You can also open issues for any bugs or feature requests.

### Steps to Contribute:
1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
