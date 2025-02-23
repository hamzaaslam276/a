# Support Vector Machines (SVM) for Handwritten Digit Classification

## Overview
This project demonstrates the implementation of Support Vector Machines (SVM) to classify handwritten digits using the [Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) from scikit-learn. The dataset consists of 8x8 pixel images representing numbers from 0 to 9.

## Dataset
The `digits` dataset contains:
- `data`: The feature matrix of shape (n_samples, n_features), where each row is a flattened 8x8 image.
- `target`: The labels (digits 0-9) corresponding to the images.
- `target_names`: Names of the classes (digits 0-9).
- `images`: The original 8x8 images.

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Code Implementation
1. Load necessary libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
```

2. Load the dataset:
```python
digits = load_digits()
```

3. Explore dataset attributes:
```python
print(dir(digits))
print(digits.target_names)
```

4. Convert dataset into a DataFrame:
```python
df = pd.DataFrame(digits.data)
df['target'] = digits.target
print(df.head())
```

## Future Enhancements
- Train an SVM classifier and evaluate its accuracy.
- Implement feature scaling for better model performance.
- Visualize digit images using `matplotlib`.
- Compare SVM with other classifiers (e.g., Random Forest, Neural Networks).

## Author
**Hamza Aslam**
