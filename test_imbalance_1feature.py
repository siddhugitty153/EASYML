
import pandas as pd
import numpy as np
import sys
import os

# Create dummy data with 1 feature
X = pd.DataFrame({'feature1': np.random.rand(100)})
y = np.array([0]*90 + [1]*10)

try:
    from imblearn.over_sampling import SMOTE
    print("Testing SMOTE with 1 feature...")
    sampler = SMOTE(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    print("SMOTE Success!")
except Exception as e:
    print(f"SMOTE Failed: {e}")

try:
    from imblearn.over_sampling import ADASYN
    print("\nTesting ADASYN with 1 feature...")
    sampler = ADASYN(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    print("ADASYN Success!")
except Exception as e:
    print(f"ADASYN Failed: {e}")
