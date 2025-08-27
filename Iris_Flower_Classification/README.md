# Iris Flower Classification 🌸

This project is part of my Data Science internship at Oasis Infobyte. It implements a machine learning model to classify iris flowers into three species based on their sepal and petal measurements.

## Project Overview

The Iris flower classification is a classic machine learning problem that involves predicting the species of iris flowers based on their morphological characteristics. This project demonstrates fundamental data science workflows including data exploration, preprocessing, model training, and evaluation.

## Dataset

The project uses the famous Iris dataset which contains:
- 150 samples (50 from each of three species)
- 4 features for each sample:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- Target variable: Species (setosa, versicolor, virginica)

## Project Structure

- `main.py` - Main script that runs the complete project
- `requirements.txt` - Python dependencies
- `data/iris.csv` - Iris dataset 
- `results/` - Folder for output visualizations
- `iris_classifier.pkl` - Saved trained model (created after running)
- `scaler.pkl` - Saved data scaler (created after running)


## Technologies Used

- Python 3.x
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Scikit-learn - Machine learning library
- Matplotlib & Seaborn - Data visualization
- Jupyter Notebook - Interactive development environment

## Machine Learning Models Implemented

The project implements and compares several classification algorithms:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Support Vector Machine (SVM)

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/trisha-sagar764/OIBSIP.git
cd OIBSIP/Iris_Flower_Classification
