# MRI Tumor Detection Project

![mri_image](https://github.com/jonagitsdata/Capstone-Final-Project-/assets/104871382/12fd121e-9602-49ec-8aa7-541cb5cc4596)


## Overview
This repository contains the code and resources for a Convolutional Neural Network (CNN) based project focused on tumor detection in MRI images. The goal of this project is to develop a deep learning model capable of accurately classifying MRI images as either having a tumor or not having a tumor.

## Dataset
The dataset used for this project consists of MRI images collected from Kaggle: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=pred . The images are accompanied by binary labels indicating whether a tumor is present or not. The dataset was preprocessed to ensure uniformity in size and format, and it was split into training, validation, and test sets.

## Model Architecture
The CNN model was designed to effectively learn spatial features from the MRI images. It includes convolutional layers, activation functions, pooling layers, and fully connected layers. The model was implemented using TensorFlow and Keras, which are popular deep learning frameworks.

## Model Training and Evaluation
The CNN model was trained on the training dataset for 20 epochs using the Adam optimizer and binary cross-entropy loss function. During training, the model's performance was monitored on the validation dataset to avoid overfitting. Various evaluation metrics, including loss, accuracy, sensitivity, specificity, precision, and F1 score, were used to assess the model's performance.

## Fine-tuning and Optimization
To optimize the model's performance, hyperparameter tuning was performed on the validation set. Fine-tuning involved adjusting the learning rate, batch size, and other parameters to achieve the best results. Data augmentation techniques were also used to increase the diversity of the training dataset and enhance the model's generalization.

## Testing and Validation
The final trained model was evaluated on the test dataset to measure its accuracy in detecting tumors. The test dataset consists of previously unseen MRI images, providing an estimate of how the model would perform on new, real-world data.

## Results
The trained CNN model demonstrated promising results in tumor detection on MRI images. The model achieved high accuracy, sensitivity, specificity, precision, and F1 score on the test dataset, indicating its effectiveness in identifying tumor cases.

## Repository Structure
- `code/`: Contains the Python scripts for data preprocessing, model training, and evaluation.
- `images/`: Contains images used in the README.
- `data/`: Includes the processed dataset split into training, validation, and test sets.
- `model/`: Contains the saved trained model for future use.
- `README.md`: The project's documentation, providing an overview of the project and its components.

## Usage
To run the code and train the CNN model, follow the steps outlined in the `JonathanMRInewCode.ipynb` file.

## Acknowledgments
I would like to acknowledge the authors of the dataset used in this project for making their data publicly available.

## References
- [[Link to paper or resource if applicable]]
- [[GitHub repository URL for the code used in this project](https://github.com/jonagitsdata/Capstone-Final-Project-)https://github.com/jonagitsdata/Capstone-Final-Project-]
