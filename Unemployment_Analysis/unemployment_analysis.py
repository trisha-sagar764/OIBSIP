# unemployment_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("husl")

# Load the datasets with error handling
try:
    df1 = pd.read_csv('Unemployment in India.csv')
    df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the CSV files are in the same directory as this script.")
    exit()

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.replace(' ', '')  # Remove internal spaces
    df.columns = df.columns.str.replace('%', '')  # Remove % signs
    df.columns = df.columns.str.replace('(', '')  # Remove parentheses
    df.columns = df.columns.str.replace(')', '')  # Remove parentheses
    return df

# Clean column names
df1 = clean_column_names(df1)
df2 = clean_column_names(df2)

# Display column names to verify
print("Columns in Dataset 1:", list(df1.columns))
print("Columns in Dataset 2:", list(df2.columns))

# Check for date columns with different names
date_columns = ['Date', 'date', 'DATE', 'Month', 'month']
date_col_df1 = None
date_col_df2 = None

for col in date_columns:
    if col in df1.columns:
        date_col_df1 = col
    if col in df2.columns:
        date_col_df2 = col

if date_col_df1 is None:
    print("Error: Could not find date column in first dataset")
    print("Available columns:", list(df1.columns))
    exit()

if date_col_df2 is None:
    print("Error: Could not find date column in second dataset")
    print("Available columns:", list(df2.columns))
    exit()

print(f"Using '{date_col_df1}' as date column for first dataset")
print(f"Using '{date_col_df2}' as date column for second dataset")

# Rename date columns to standard name
df1 = df1.rename(columns={date_col_df1: 'Date'})
df2 = df2.rename(columns={date_col_df2: 'Date'})

# Check for missing values
print("\nMissing values in Dataset 1:")
print(df1.isnull().sum())
print("\nMissing values in Dataset 2:")
print(df2.isnull().sum())

# Clean the data - remove rows with missing values
df1_clean = df1.dropna()
df2_clean = df2.dropna()

# Convert Date column to datetime format with error handling
try:
    df1_clean['Date'] = pd.to_datetime(df1_clean['Date'])
except Exception as e:
    print(f"Error converting dates in first dataset: {e}")
    print("Sample dates:", df1_clean['Date'].head().tolist())
    # Try alternative date parsing
    df1_clean['Date'] = pd.to_datetime(df1_clean['Date'], errors='coerce')
    
try:
    df2_clean['Date'] = pd.to_datetime(df2_clean['Date'])
except Exception as e:
    print(f"Error converting dates in second dataset: {e}")
    print("Sample dates:", df2_clean['Date'].head().tolist())
    # Try alternative date parsing
    df2_clean['Date'] = pd.to_datetime(df2_clean['Date'], errors='coerce')

# Remove any rows where date conversion failed
df1_clean = df1_clean.dropna(subset=['Date'])
df2_clean = df2_clean.dropna(subset=['Date'])

# Check unique values in Region column
print("\nUnique regions in Dataset 1:")
print(df1_clean['Region'].unique())
print("\nUnique regions in Dataset 2:")
print(df2_clean['Region'].unique())

# Standardize region names if needed
df1_clean['Region'] = df1_clean['Region'].str.strip()
df2_clean['Region'] = df2_clean['Region'].str.strip()

# Combine both datasets for comprehensive analysis
# First, ensure both have the same columns
df1_clean = df1_clean[['Region', 'Date', 'EstimatedUnemploymentRate', 
                      'EstimatedEmployed', 'EstimatedLabourParticipationRate', 'Area']]

df2_clean = df2_clean[['Region', 'Date', 'EstimatedUnemploymentRate', 
                      'EstimatedEmployed', 'EstimatedLabourParticipationRate']]

# Add Area column to df2_clean (assuming based on region)
region_to_area = {
    'AndhraPradesh': 'South',
    'Assam': 'Northeast',
    'Bihar': 'East',
    'Chhattisgarh': 'Central',
    'Delhi': 'North',
    'Goa': 'West',
    'Gujarat': 'West',
    'Haryana': 'North',
    'HimachalPradesh': 'North',
    'Jammu&Kashmir': 'North',
    'Jharkhand': 'East',
    'Karnataka': 'South',
    'Kerala': 'South',
    'MadhyaPradesh': 'Central',
    'Maharashtra': 'West',
    'Meghalaya': 'Northeast',
    'Odisha': 'East',
    'Puducherry': 'South',
    'Punjab': 'North',
    'Rajasthan': 'North',
    'Sikkim': 'Northeast',
    'TamilNadu': 'South',
    'Telangana': 'South',
    'Tripura': 'Northeast',
    'UttarPradesh': 'Central',
    'Uttarakhand': 'North',
    'WestBengal': 'East'
}

df2_clean['Area'] = df2_clean['Region'].map(region_to_area)

# Combine both datasets
combined_df = pd.concat([df1_clean, df2_clean], ignore_index=True)

# Remove duplicates if any
combined_df = combined_df.drop_duplicates()

# Analysis 1: Overall unemployment trend over time
plt.figure(figsize=(14, 8))
monthly_avg = combined_df.groupby(pd.Grouper(key='Date', freq='M'))['EstimatedUnemploymentRate'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
plt.title('Average Unemployment Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('unemployment_trend.png')
plt.show()

# Analysis 2: Unemployment by Region
plt.figure(figsize=(14, 10))
region_avg = combined_df.groupby('Region')['EstimatedUnemploymentRate'].mean().sort_values(ascending=False)
region_avg.plot(kind='bar')
plt.title('Average Unemployment Rate by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('unemployment_by_region.png')
plt.show()

# Analysis 3: Unemployment by Area
plt.figure(figsize=(12, 8))
area_avg = combined_df.groupby('Area')['EstimatedUnemploymentRate'].mean().sort_values(ascending=False)
area_avg.plot(kind='bar')
plt.title('Average Unemployment Rate by Area', fontsize=16)
plt.xlabel('Area', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('unemployment_by_area.png')
plt.show()

# Analysis 4: Employment vs Unemployment correlation
plt.figure(figsize=(10, 8))
sns.scatterplot(data=combined_df, x='EstimatedEmployed', y='EstimatedUnemploymentRate', hue='Area')
plt.title('Employment vs Unemployment Rate', fontsize=16)
plt.tight_layout()
plt.savefig('employment_vs_unemployment.png')
plt.show()

# Analysis 5: Labour Participation Rate vs Unemployment Rate
plt.figure(figsize=(10, 8))
sns.scatterplot(data=combined_df, x='EstimatedLabourParticipationRate', 
                y='EstimatedUnemploymentRate', hue='Area')
plt.title('Labour Participation Rate vs Unemployment Rate', fontsize=16)
plt.tight_layout()
plt.savefig('participation_vs_unemployment.png')
plt.show()

# Analysis 6: COVID-19 impact comparison (before and after March 2020)
pre_covid = combined_df[combined_df['Date'] < pd.Timestamp(2020, 3, 1)]
post_covid = combined_df[combined_df['Date'] >= pd.Timestamp(2020, 3, 1)]

pre_covid_avg = pre_covid['EstimatedUnemploymentRate'].mean()
post_covid_avg = post_covid['EstimatedUnemploymentRate'].mean()

print(f"\nAverage Unemployment Rate Before COVID-19 (Before March 2020): {pre_covid_avg:.2f}%")
print(f"Average Unemployment Rate After COVID-19 (After March 2020): {post_covid_avg:.2f}%")
print(f"Percentage Increase: {((post_covid_avg - pre_covid_avg) / pre_covid_avg * 100):.2f}%")

# Visualization of COVID impact
plt.figure(figsize=(10, 6))
periods = ['Pre-COVID', 'Post-COVID']
values = [pre_covid_avg, post_covid_avg]
plt.bar(periods, values, color=['blue', 'red'])
plt.title('Unemployment Rate: Pre vs Post COVID-19', fontsize=16)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.tight_layout()
plt.savefig('covid_impact.png')
plt.show()

# Analysis 7: Top 5 regions with highest unemployment
top_5_regions = combined_df.groupby('Region')['EstimatedUnemploymentRate'].mean().nlargest(5)
print("\nTop 5 Regions with Highest Unemployment Rates:")
for region, rate in top_5_regions.items():
    print(f"{region}: {rate:.2f}%")

# Analysis 8: Bottom 5 regions with lowest unemployment
bottom_5_regions = combined_df.groupby('Region')['EstimatedUnemploymentRate'].mean().nsmallest(5)
print("\nBottom 5 Regions with Lowest Unemployment Rates:")
for region, rate in bottom_5_regions.items():
    print(f"{region}: {rate:.2f}%")

# Save the cleaned and combined dataset
combined_df.to_csv('combined_unemployment_data.csv', index=False)
print("\nCleaned and combined dataset saved as 'combined_unemployment_data.csv'")
print("\nAll visualizations have been generated and saved as PNG files!")