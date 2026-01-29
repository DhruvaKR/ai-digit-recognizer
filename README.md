🧠 AI Handwritten Digit Recognition using Multi-Layer Perceptron (MLP)

A deep learning-based handwritten digit recognition system built using Multi-Layer Perceptron (MLP) and deployed as an interactive web application using Streamlit. The model achieves 98% accuracy on the MNIST dataset and allows real-time digit prediction through drawing and image upload.

📌 Project Overview

This project focuses on designing, training, evaluating, and deploying a Multi-Layer Perceptron (MLP) neural network for handwritten digit classification using the MNIST dataset.

The project demonstrates the complete machine learning pipeline, starting from data preprocessing and model training to evaluation and real-world deployment using a professional-grade web interface. The final system allows users to draw digits or upload images and instantly receive predictions with confidence scores.

This project highlights strong skills in:

Machine Learning

Deep Learning

Model Evaluation

Web Deployment

User Experience (UX) design

End-to-end AI system development

🎯 Objectives

To build an accurate MLP neural network model for digit classification.

To understand the effect of hidden layers and activation functions on model performance.

To evaluate the model using accuracy, loss, confusion matrix, and classification report.

To compare MLP performance with baseline machine learning models.

To deploy the trained model as a real-time interactive web application.

To create a portfolio-quality AI project demonstrating both ML and deployment skills.

🧩 Problem Statement

Handwritten digit recognition is a classical and fundamental problem in computer vision, widely used in applications such as:

Postal mail sorting

Bank cheque processing

Digital form reading

Optical Character Recognition (OCR)

Traditional machine learning algorithms struggle to achieve high accuracy due to the high dimensionality of image data. Therefore, a deep learning model capable of learning complex non-linear patterns is required.

This project solves this problem using a Multi-Layer Perceptron (MLP) trained on pixel-level data from the MNIST dataset.

📊 Dataset Description

MNIST Handwritten Digit Dataset

Total images: 70,000

Training images: 60,000

Testing images: 10,000

Image size: 28 × 28 pixels

Color format: Grayscale

Output classes: 10 (digits 0–9)

The dataset is automatically loaded using TensorFlow’s built-in dataset API.

⚙️ Technologies & Tools Used

Programming Language: Python

Deep Learning Framework: TensorFlow (Keras API)

Web Framework: Streamlit

Libraries: NumPy, Matplotlib, Pillow

Development Tools: Google Colab, VS Code, GitHub

🔍 Methodology & Workflow
1️⃣ Data Preprocessing

Loaded the MNIST dataset using TensorFlow API.

Normalized pixel values from 0–255 → 0–1 for faster convergence.

Flattened 28×28 images into 784-dimensional vectors.

Performed one-hot encoding of labels.

2️⃣ Model Architecture (MLP)

The neural network consists of:

Input Layer  → 784 neurons  
Hidden Layer 1 → 128 neurons (ReLU activation)  
Hidden Layer 2 → 64 neurons (ReLU activation)  
Output Layer → 10 neurons (Softmax activation)


Activation Functions: ReLU, Softmax

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

3️⃣ Model Training

Epochs: 20

Batch Size: 128

Validation Split: 10%

Achieved stable convergence with high accuracy.

4️⃣ Model Evaluation

Performance metrics used:

Accuracy

Loss

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📌 Final Test Accuracy: 98%

5️⃣ Model Comparison

The MLP model was compared with baseline algorithms:

Model	Accuracy
Logistic Regression	~92%
k-Nearest Neighbors (k-NN)	~96%
MLP (Proposed Model)	~98%

📈 MLP outperformed classical models due to its ability to learn complex non-linear features.

🌐 Web Application Deployment

The trained model was deployed using Streamlit to create a professional interactive web application.

Features:

✏ Real-time digit drawing

📤 Image upload prediction

🎯 Instant prediction results

📊 Confidence score display

🔍 Top-3 class predictions

🎨 Modern light UI with purple gradient

⚡ Ultra-fast prediction using model caching

🖥️ How to Run Locally
Step 1: Clone Repository
git clone https://github.com/DhruvaKR/ai-digit-recognizer.git
cd ai-digit-recognizer

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Web App
streamlit run app.py

Open browser at:
http://localhost:8501

🏆 Key Results

Achieved 98% accuracy using a pure MLP model (without CNN).

Built a full-stack AI system from training to deployment.

Developed a professional-grade UI for real-time predictions.

Successfully deployed the model as a live web application.
