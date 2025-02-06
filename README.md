# **Potato Leaf Disease Detection**

This repository contains a machine learning-based model for detecting diseases in potato leaves. It uses image classification to identify various potato leaf diseases, aiding farmers in disease management and early detection.

## **Table of Contents**
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)

## **Introduction**

This project aims to automatically detect diseases in potato leaves using deep learning. The model can classify the condition of potato leaves into different disease categories, including healthy leaves.

## **Technologies Used**

- **Python**: Programming language for developing the project.
- **TensorFlow/Keras**: Deep learning framework for building the disease detection model.
- **Flask**: Web framework used in the `app.py` file to serve the model for inference.
- **OpenCV**: For image processing and augmentation.
- **Matplotlib/Seaborn**: For visualizing training metrics.
- **Pandas/Numpy**: For data manipulation.

## **Installation Instructions**

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**  
   Clone this repository using Git:
   ```bash
   git clone https://github.com/VedantBende/potato-leaf-disease-detection.git
   ```

2. **Create a Virtual Environment**  
   It's recommended to create a virtual environment:
   ```bash
   python -m venv env
   ```

3. **Activate the Virtual Environment**  
   On Windows:
   ```bash
   .\env\Scripts\activate
   ```
   On macOS/Linux:
   ```bash
   source env/bin/activate
   ```

4. **Install Dependencies**  
   Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

## **Dataset**

The dataset used in this project contains images of potato leaves affected by various diseases. Ensure your dataset is well-organized in folders representing different diseases (e.g., "Healthy", "Early Blight", "Late Blight").

## **Usage**

1. **Running the Flask Web App**  
   To serve the model through a web app, use the following command to run `app.py`:
   ```bash
   python app.py
   ```

   This will start the web application, allowing you to upload an image of a potato leaf and receive a prediction.

2. **Prediction**  
   Once the app is running, visit `http://127.0.0.1:5000/` in your browser. You can upload potato leaf images to get disease predictions.

## **Model Training**

1. **Prepare the Dataset**  
   Ensure the dataset is well-organized with labeled folders for each disease type.

2. **Run the Jupyter Notebook for Training**  
   Open `Train_potato_disease.ipynb` in Jupyter Notebook or Jupyter Lab. This notebook contains the steps to load the dataset, preprocess the images, and train the model.

   Run all the cells to train the model. The model will be saved as `trained_plant_disease_model.keras` after training.

3. **Model Checkpoint**  
   The final trained model will be saved in the file `trained_plant_disease_model.keras`, which can be used for inference.

## **Results**

You can visualize the model's training results (e.g., accuracy and loss) using the plots generated in the Jupyter Notebook during training. The notebook will also display any evaluation metrics used for model performance.
