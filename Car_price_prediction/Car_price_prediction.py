# Car Price Prediction with Machine Learning - Fixed Version
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the CSV data
def load_data():
    df = pd.read_csv('car data.csv')
    return df

# Simplified preprocessing (removing problematic features)
def preprocess_data(df):
    # Create a copy
    df_processed = df.copy()
    
    # Calculate car age (assuming current year is 2024)
    current_year = 2024
    df_processed['Car_Age'] = current_year - df_processed['Year']
    
    # Encode categorical variables
    le_fuel = LabelEncoder()
    le_transmission = LabelEncoder()
    le_selling_type = LabelEncoder()
    
    df_processed['Fuel_Type_Encoded'] = le_fuel.fit_transform(df_processed['Fuel_Type'])
    df_processed['Transmission_Encoded'] = le_transmission.fit_transform(df_processed['Transmission'])
    df_processed['Selling_type_Encoded'] = le_selling_type.fit_transform(df_processed['Selling_type'])
    
    # Select only reliable features (remove Price_Ratio as it causes data leakage)
    features = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age', 
                'Fuel_Type_Encoded', 'Transmission_Encoded', 'Selling_type_Encoded']
    
    X = df_processed[features]
    y = df_processed['Selling_Price']
    
    return X, y, df_processed, le_fuel, le_transmission, le_selling_type

# Enhanced EDA with more insights
def perform_eda(df):
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"\nTotal Cars: {len(df)}")
    print(f"Time Range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Price Range: Rs. {df['Selling_Price'].min():.2f} - Rs. {df['Selling_Price'].max():.2f} lakhs")
    
    # Top 10 car models by frequency
    print(f"\nTop 10 Car Models:")
    top_cars = df['Car_Name'].value_counts().head(10)
    for car, count in top_cars.items():
        print(f"  {car}: {count} cars")
    
    # Price analysis by fuel type
    print(f"\nAverage Price by Fuel Type:")
    fuel_stats = df.groupby('Fuel_Type')['Selling_Price'].agg(['mean', 'count'])
    for fuel_type, stats in fuel_stats.iterrows():
        print(f"  {fuel_type}: Rs. {stats['mean']:.2f} lakhs ({int(stats['count'])} cars)")
    
    # Price analysis by transmission
    print(f"\nAverage Price by Transmission:")
    trans_stats = df.groupby('Transmission')['Selling_Price'].agg(['mean', 'count'])
    for trans, stats in trans_stats.iterrows():
        print(f"  {trans}: Rs. {stats['mean']:.2f} lakhs ({int(stats['count'])} cars)")
    
    # Price analysis by owner type
    print(f"\nAverage Price by Number of Owners:")
    owner_stats = df.groupby('Owner')['Selling_Price'].agg(['mean', 'count'])
    for owner, stats in owner_stats.iterrows():
        print(f"  {int(owner)} owner(s): Rs. {stats['mean']:.2f} lakhs ({int(stats['count'])} cars)")

# Model training with proper validation
def train_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=10)
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"  MAE: Rs. {mae:.3f} lakhs")
        print(f"  RMSE: Rs. {rmse:.3f} lakhs")
        print(f"  R2 Score: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
    
    return results, X_train, X_test, y_train, y_test, best_model, best_model_name

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        for _, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        return importance_df
    return None

# Prediction function
def predict_car_price(model, present_price, driven_kms, owner, car_age, 
                     fuel_type, transmission, selling_type, le_fuel, le_transmission, le_selling_type):
    
    # Encode categorical variables
    try:
        fuel_encoded = le_fuel.transform([fuel_type])[0]
    except:
        fuel_encoded = 0
    
    try:
        transmission_encoded = le_transmission.transform([transmission])[0]
    except:
        transmission_encoded = 0
    
    try:
        selling_type_encoded = le_selling_type.transform([selling_type])[0]
    except:
        selling_type_encoded = 0
    
    features = np.array([[
        present_price, driven_kms, owner, car_age,
        fuel_encoded, transmission_encoded, selling_type_encoded
    ]])
    
    prediction = model.predict(features)
    return prediction[0]

# Generate detailed report
def generate_report(y_test, y_pred, model_name):
    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE REPORT")
    print("=" * 60)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Model: {model_name}")
    print(f"Mean Absolute Error: Rs. {mae:.3f} lakhs")
    print(f"Root Mean Squared Error: Rs. {rmse:.3f} lakhs")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy Percentage: {(1 - (mae / y_test.mean())) * 100:.1f}%")
    
    # Error distribution analysis
    errors = np.abs(y_test - y_pred)
    print(f"\nError Distribution:")
    print(f"  Min Error: Rs. {errors.min():.3f} lakhs")
    print(f"  Max Error: Rs. {errors.max():.3f} lakhs")
    print(f"  Median Error: Rs. {np.median(errors):.3f} lakhs")
    print(f"  95% of errors below: Rs. {np.percentile(errors, 95):.3f} lakhs")

# Main execution
def main():
    # Load data
    print("Loading and analyzing car data...")
    df = load_data()
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    X, y, df_processed, le_fuel, le_transmission, le_selling_type = preprocess_data(df)
    
    # Train models
    results, X_train, X_test, y_train, y_test, best_model, best_model_name = train_models(X, y)
    
    # Get predictions from best model
    y_pred = best_model.predict(X_test)
    
    # Generate detailed report
    generate_report(y_test, y_pred, best_model_name)
    
    # Analyze feature importance
    feature_names = X.columns.tolist()
    analyze_feature_importance(best_model, feature_names)
    
    # Sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    samples = [
        {'present_price': 8.5, 'driven_kms': 25000, 'owner': 0, 'car_age': 3, 
         'fuel_type': 'Petrol', 'transmission': 'Manual', 'selling_type': 'Dealer',
         'description': 'Mid-range sedan with low mileage'},
        
        {'present_price': 12.0, 'driven_kms': 15000, 'owner': 0, 'car_age': 2, 
         'fuel_type': 'Diesel', 'transmission': 'Automatic', 'selling_type': 'Dealer',
         'description': 'Premium SUV with very low mileage'},
        
        {'present_price': 5.5, 'driven_kms': 45000, 'owner': 0, 'car_age': 5, 
         'fuel_type': 'Petrol', 'transmission': 'Manual', 'selling_type': 'Individual',
         'description': 'Economy hatchback with medium mileage'},
        
        {'present_price': 20.0, 'driven_kms': 8000, 'owner': 0, 'car_age': 1, 
         'fuel_type': 'Diesel', 'transmission': 'Automatic', 'selling_type': 'Dealer',
         'description': 'Luxury SUV with very low mileage'},
        
        {'present_price': 3.2, 'driven_kms': 75000, 'owner': 1, 'car_age': 8, 
         'fuel_type': 'Petrol', 'transmission': 'Manual', 'selling_type': 'Individual',
         'description': 'Older car with high mileage and one previous owner'}
    ]
    
    for i, sample in enumerate(samples, 1):
        pred = predict_car_price(
            best_model,
            sample['present_price'],
            sample['driven_kms'],
            sample['owner'],
            sample['car_age'],
            sample['fuel_type'],
            sample['transmission'],
            sample['selling_type'],
            le_fuel, le_transmission, le_selling_type
        )
        print(f"\nSample {i}: {sample['description']}")
        print(f"  Present Price: Rs. {sample['present_price']:.1f} lakhs")
        print(f"  Predicted Selling Price: Rs. {pred:.2f} lakhs")
        print(f"  Expected Depreciation: Rs. {sample['present_price'] - pred:.2f} lakhs")
        print(f"  Depreciation Percentage: {((sample['present_price'] - pred) / sample['present_price'] * 100):.1f}%")
    
    print("\n" + "=" * 60)
    print("MODEL INSIGHTS")
    print("=" * 60)
    print("* Random Forest typically performs best for this type of data")
    print("* Present Price is usually the most important feature")
    print("* Car Age and Mileage significantly affect resale value")
    print("* Automatic transmission cars generally have higher resale values")
    print("* Diesel cars maintain better value than petrol cars")
    print("* Cars sold through dealers get better prices than individual sales")
    print("\n" + "=" * 60)
    print("CAR PRICE PREDICTION MODEL READY!")
    print("=" * 60)

if __name__ == "__main__":
    main()