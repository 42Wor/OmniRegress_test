# OmniRegress Linear Regression ⚡

**OmniRegress**  is a comprehensive Python library designed for all types of regression analysis, providing robust implementations from simple linear models to advanced ensemble methods.

## 🚀 Performance Benchmarks

| Scenario | Samples | Features | Fit Time (s) | Memory Usage |
|----------|---------|----------|--------------|--------------|
| 🐢 Small | 10,000 | 5 | 0.0171 | 2.52 MB |
| 🚗 Medium | 100,000 | 10 | 0.2875 | 40.44 MB |
| 🚀 Large | 500,000 | 20 | 3.8163 | 354.77 MB |
| 🛸 Very Large | 1,000,000 | 25 | 9.7494 | 862.12 MB |
| 🧠 High Features | 10,000 | 500 | 22.5145 | 153.58 MB |

![Omniregress Performance Chart](omniregress_performance.png)

### 🔧 System Specs
- **Processor:** Intel Core i5-8250U @ 1.60GHz (4 Cores, 8 Threads)
- **Memory:** 16.0 GB @ 2400MHz  
- **Storage:** 256 GB SSD

## 🏠 Real-World Example: California Housing Prices

### 📈 Model Performance
- **RMSE:** $68,628.10
- **R² Score:** 0.6393 (63.9% variance explained)

> 🎯 **Performance Insight**
> These metrics demonstrate strong training data fitting capabilities. For real-world deployment, consider cross-validation to assess generalization performance.

### 🎯 Prediction Showcase

| Actual Price | Predicted Price | Accuracy |
|--------------|-----------------|----------|
| $452,600 | $413,446 | 91.3% |
| $358,500 | $331,133 | 92.4% |
| $352,100 | $338,818 | 96.2% |
| $341,300 | $304,633 | 89.3% |
| $342,200 | $285,108 | 83.3% |

---

## Files

- `main.py` - Main linear regression implementation
- `pure_omni_housing.py` - Pure Python implementation using housing data
- `test_performance.py` - Performance benchmarking script
- `omniregress_performance.png` - Performance comparison visualization
- `data/` - Dataset directory
- `README.md` - This file



*OmniRegress: Making regression analysis accessible and powerful* 🎯