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
- `JonathanMRInewCode.ipynb/`: Contains the Python scripts for data preprocessing, model training, and evaluation.
- `dataset2, prediction and extra validation set/`: Contains images used for MRI tumor detection.
- `jpg files/`: used for model testing.
- `Nkangabwa_Capstone_Project_FinalReport.pdf/`: Final report detailing the project and results
- `models and finalmodel/`: Contains the saved trained model for future use.
- `README.md`: The project's documentation, providing an overview of the project and its components.

## Usage
To run the code and train the CNN model, follow the steps outlined in the `JonathanMRInewCode.ipynb` file.

## Acknowledgments
I would like to acknowledge the authors of the dataset used in this project for making their data publicly available.

## References


1. Arabahmadi, M., Farahbakhsh, R., Rezazadeh, J.: Deep learning for smart healthcaremdash;a survey on brain tumor detection from medical imaging. Sensors 22(5) (2022). https://doi.org/10.3390/s22051960, https://www.mdpi.com/14248220/22/5/1960

2. Khan MSI, Rahman A, D.T.K.M.N.M.B.S.M.A.D.I.: Accurate brain tumor detection using deep convolutional neural network, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9468505/

3. Koh, D.M., Papanikolaou, N., Bick, U., et al.: Artificial intelligence and machine learning in cancer imaging. Communications Medicine 2, 133 (2022). https://doi.org/10.1038/s43856-022-00199-0

4. Mahmud, M.I., Mamun, M., Abdelgawad, A.: A deep analysis of brain tumor detection from mr images using deep learning networks. Algorithms 16(4) (2023). https://doi.org/10.3390/a16040176, https://www.mdpi.com/1999-4893/16/4/176

5. Mostafa AM, Zakariah M, A.E.: Brain tumor segmentation using deep learning on mri images, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10177460/sec3diagnostics-13-01562title

6. Nkangabwa, J.: Capstone final project - mri tumor detection.

https://github.com/jonagitsdata/Capstone-Final-Project- (2023), accessed:

7.01.2023

7. Nothnagel, N.: Creating deep learning image classifiers - nick nothnagel. https://www.youtube.com/watch?v=jztwpsIzEGc (2022), accessed: 7.15.2023

8. Nothnagel, N.: Image classification using tensorflow 2.x and keras.

https://github.com/nicknochnack/ImageClassification/blob/main/Getting20Started.ipynb (2022), accessed: 7.15.2023 10

J. Nkangabwa.

9. Rikiya Yamashita, Mizuho Nishio, R.K.G.D.K.T.: Convolutional neural networks: an overview and application in radiology, https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-06399citeas

10. Taher, F., Shoaib, M.R., Emara, H.M., Abdelwahab, K.M., Abd El-Samie, F., Haweel, M.T.: Efficient framework for brain tumor detection using different deep learning techniques. Frontiers in Public Health 10, 959667 (2022). https://doi.org/10.3389/fpubh.2022.959667

11. Tan, M., Le, Q.V.: Efficientnet: Rethinking model scaling for convolutional neural networks. CoRR abs/1905.11946 (2019), http://arxiv.org/abs/1905.11946
