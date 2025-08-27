# Iris Flower Classification - Final Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def create_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Created data and results directories")

def load_and_explore_data():
    """Load and explore the iris dataset"""
    print("=== LOADING AND EXPLORING DATA ===")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # Save the dataset to CSV for reference
    df.to_csv('data/iris.csv', index=False)
    print("Dataset saved as 'data/iris.csv'")
    
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nStatistical summary:")
    print(df.describe())
    print("\nSpecies distribution:")
    print(df['species'].value_counts())
    
    return df, iris.feature_names, iris.target_names

def visualize_data(df, feature_names, target_names):
    """Create visualizations of the data"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set style for better looking plots
    plt.style.use('default')
    
    # Box plots for each feature
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Feature Distribution by Species', fontsize=16)
    
    for i, feature in enumerate(feature_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Create boxplot for each species
        data_by_species = [df[df['species'] == species][feature] for species in target_names]
        box_plot = ax.boxplot(data_by_species, tick_labels=target_names, patch_artist=True)
        
        # Add colors to boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_ylabel('cm')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/feature_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.drop(['target', 'species'], axis=1)
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Correlation Heatmap', fontweight='bold')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text_color = 'white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold',
                    color=text_color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to results folder")

def prepare_data(df):
    """Prepare data for training"""
    print("\n=== PREPARING DATA ===")
    X = df.drop(['target', 'species'], axis=1).values
    y = df['target'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names):
    """Train multiple models and evaluate them"""
    print("\n=== TRAINING MODELS ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': accuracy}
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def evaluate_best_model(results, X_test, y_test, target_names, feature_names):
    """Evaluate the best performing model"""
    print("\n=== EVALUATING BEST MODEL ===")
    
    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Save the best model
    with open('iris_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved as 'iris_classifier.pkl'")
    
    # Detailed evaluation
    y_pred = best_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(len(target_names)), target_names, rotation=45)
    plt.yticks(range(len(target_names)), target_names)
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add values to cells
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            plt.text(j, i, str(cm[i, j]), 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=12,
                    color=text_color)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance', fontweight='bold')
        plt.xlabel('Importance')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return best_model

def make_predictions(model, feature_names, target_names):
    """Make predictions on new data"""
    print("\n=== MAKING PREDICTIONS ===")
    
    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Example predictions
    examples = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.0, 2.7, 5.1, 1.6],  # virginica
        [5.5, 2.4, 3.8, 1.1]   # versicolor
    ]
    
    print("Example predictions:")
    print("-" * 50)
    
    for i, example in enumerate(examples):
        # Scale the example
        example_scaled = scaler.transform([example])
        
        # Make prediction
        prediction = model.predict(example_scaled)
        probability = model.predict_proba(example_scaled)
        
        species = target_names[prediction[0]]
        confidence = np.max(probability) * 100
        
        print(f"Example {i+1}:")
        print(f"  Measurements: {example}")
        print(f"  Predicted: {species}")
        print(f"  Confidence: {confidence:.2f}%")
        print()

def main():
    """Main function to run the complete project"""
    print("IRIS FLOWER CLASSIFICATION PROJECT")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Load and explore data
    df, feature_names, target_names = load_and_explore_data()
    
    # Visualize data
    visualize_data(df, feature_names, target_names)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names)
    
    # Evaluate best model - FIXED: added feature_names parameter
    best_model = evaluate_best_model(results, X_test, y_test, target_names, feature_names)
    
    # Make predictions
    make_predictions(best_model, feature_names, target_names)
    
    print("\n=== PROJECT COMPLETE ===")
    print("Dataset saved: data/iris.csv")
    print("Visualizations saved: results/ folder")
    print("Models saved: iris_classifier.pkl, scaler.pkl")
    print("=" * 50)

if __name__ == "__main__":
    main()