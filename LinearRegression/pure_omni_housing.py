import pandas as pd
import numpy as np

# Import your custom Linear Regression model
from omniregress import LinearRegression

# --- Helper Functions (Not used for splitting, but kept for metrics) ---

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared (R²) score."""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# --- Main Script ---

print("--- California Housing Price Prediction (Training on Full Dataset) ---")

# --- 1. Load the Data ---
try:
    df = pd.read_csv('data/housing.csv')
    print("✅ Step 1: Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'housing.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- 2. Simple Preprocessing ---
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
print("✅ Step 2: Preprocessing complete.")


# --- 3. Prepare Data for Modeling ---
# We will use the entire dataset for both training and evaluation
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
print(f"✅ Step 3: Prepared full dataset with {len(X)} samples for training and evaluation.")


# --- 4. Train the Linear Regression Model on the Full Dataset ---
model = LinearRegression()
model.fit(X.to_numpy(), y.to_numpy())
print("✅ Step 4: Model training on the entire dataset is complete.")


# --- 5. Evaluate the Model on the Same Full Dataset ---
predictions = model.predict(X.to_numpy())

# Calculate metrics by comparing the original y values with the predictions
rmse = calculate_rmse(y.to_numpy(), predictions)
r2 = calculate_r2(y.to_numpy(), predictions)

print(f"\nRoot Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"\nInterpretation:")
print(f" -> The model fits the training data with a typical error of about ${rmse:,.2f}.")
print(f" -> The model explains {r2:.1%} of the variance in the training data.")


# --- 6. Show Example Predictions ---
print("\n--- 🔍 Example Predictions ---")
results = pd.DataFrame({
    'Actual Value': y.values, 
    'Predicted Value': predictions
})
results['Difference'] = results['Actual Value'] - results['Predicted Value']

print(results.head().to_string(formatters={'Actual Value': '${:,.0f}'.format,
                                           'Predicted Value': '${:,.0f}'.format,
                                           'Difference': '${:,.0f}'.format}))