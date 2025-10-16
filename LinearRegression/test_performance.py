import time
import tracemalloc
import numpy as np
import pandas as pd

# Import your model
from omniregress import LinearRegression as OmniLinearRegression

def generate_data(n_samples, n_features):
    """Generates synthetic data for a regression problem."""
    X = np.random.rand(n_samples, n_features)
    
    # Create a true linear relationship for y
    true_coefficients = np.random.rand(n_features) * 10
    true_intercept = 5
    noise = np.random.randn(n_samples) * 0.5
    y = np.dot(X, true_coefficients) + true_intercept + noise
    
    return X, y

def run_performance_test(model_class, n_samples, n_features):
    """
    Runs a single performance test for a given model and data size.
    
    Returns:
        tuple: (fit_time, predict_time, peak_memory_mb)
    """
    X, y = generate_data(n_samples, n_features)
    model = model_class()
    
    # --- Time Measurement ---
    start_fit = time.perf_counter()
    model.fit(X, y)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit
    
    start_predict = time.perf_counter()
    model.predict(X)
    end_predict = time.perf_counter()
    predict_time = end_predict - start_predict

    # --- Memory Measurement ---
    tracemalloc.start()
    
    # Run the operation again to measure memory
    model_for_mem_test = model_class()
    model_for_mem_test.fit(X, y)
    model_for_mem_test.predict(X)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak memory usage in Megabytes
    peak_memory_mb = peak / 1024 / 1024
    
    return fit_time, predict_time, peak_memory_mb

if __name__ == "__main__":
    # --- Define Test Scenarios ---
    # List of (scenario_name, n_samples, n_features)
    scenarios = [
        ("Small", 10_000, 5),
        ("Medium", 100_000, 10),
        ("Large", 500_000, 20),
        ("Very Large", 1_000_000, 25),
        ("High Features", 10_000, 500), # Tests performance with many features
    ]
    
    # --- Run Tests and Collect Results ---
    results = []
    
    print("Starting Omniregress Linear Regression Performance Test...")
    
    for name, n_samples, n_features in scenarios:
        print(f"\n--- Running Scenario: {name} ({n_samples:,} samples, {n_features} features) ---")
        try:
            fit_time, predict_time, peak_mem = run_performance_test(OmniLinearRegression, n_samples, n_features)
            results.append({
                "Scenario": name,
                "Samples": f"{n_samples:,}",
                "Features": n_features,
                "Fit Time (s)": fit_time,
                "Predict Time (s)": predict_time,
                "Peak Memory (MB)": peak_mem,
            })
            print(f"  Fit Time: {fit_time:.4f}s | Predict Time: {predict_time:.4f}s | Peak Memory: {peak_mem:.2f} MB")
        except Exception as e:
            print(f"  ERROR testing scenario {name}: {e}")
            results.append({
                "Scenario": name,
                "Samples": f"{n_samples:,}",
                "Features": n_features,
                "Fit Time (s)": "Error",
                "Predict Time (s)": "Error",
                "Peak Memory (MB)": "Error",
            })

    # --- Display Results in a Table ---
    if results:
        df = pd.DataFrame(results)
        print("\n\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        # Set display options to show all columns and format numbers
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 10)
        df['Fit Time (s)'] = df['Fit Time (s)'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        df['Predict Time (s)'] = df['Predict Time (s)'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        df['Peak Memory (MB)'] = df['Peak Memory (MB)'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        print(df)
        print("="*80)