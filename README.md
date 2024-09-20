# Feature Engineering Techniques

## Table of Contents
1. [Feature Scaling](#feature-scaling)
2. [Label Encoding](#label-encoding)
3. [One-Hot Encoding](#one-hot-encoding)
4. [Outlier Handling](#outlier-handling).


---

## Feature Scaling

### Overview
Feature scaling is a crucial preprocessing step that helps normalize data within a specific range. This ensures that algorithms that rely on distance metrics (like KNN) don't get biased toward features with larger magnitudes. There are two common methods of feature scaling:
1. **Standardization (Z-Score Scaling)** - Transforms data to have a mean of 0 and a standard deviation of 1.
2. **Normalization (Min-Max Scaling)** - Scales the data between a specified range, usually [0, 1].

### Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Normalization
scaler = MinMaxScaler()
x_train_normalized = scaler.fit_transform(x_train)
```

For more detailed examples and explanations, refer to the full documentation in `Feature_Scaling.ipynb`【17†source】.

---

## Label Encoding

### Overview
Label Encoding is used to convert categorical variables into numerical values. It's typically applied when the categorical variable is ordinal or when there's no notion of rank but the algorithm requires numerical inputs.

### Example
Given a dataset with screen sizes:
```python
data = {'Screen': ['Big', 'Medium', 'Small']}
df['Screen'] = df['Screen'].replace({'Small': 0, 'Medium': 1, 'Big': 2})
```

This will replace the categorical values with numeric representations, allowing the model to interpret the data. For more details on Label Encoding, including its implementation using real-world datasets, see `Label_Encoding.ipynb`.

---

## One-Hot Encoding

### Overview
One-Hot Encoding transforms categorical variables into binary vectors, creating a new binary feature for each category. It’s most useful for nominal (non-ordinal) categorical features where no ordering exists among the categories.

### Example
```python
import pandas as pd
pd.get_dummies(df['Gender'], prefix='Gender')
```
This creates separate binary columns for each gender category (e.g., `Gender_Female`, `Gender_Male`), avoiding misleading ordinal relationships.

Alternatively, using `scikit-learn`'s `OneHotEncoder`:
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['Gender']]).toarray()
```

For detailed code examples and explanations, see `One_Hot_Encoding.ipynb`.

---

## Outlier Handling 
### Outlier Removal 
 Outliers can be removed using 2 methods :
- IQR (Inter quartile range)
- Z-score (Standar score)

### IQR 
```python

q1 = df['Coapplicant_Income'].quantile(0.25)
q3 = df['Coapplicant_Income'].quantile(0.75)
IQR = q3-q1
min_range  = q1 - (1.5*IQR)
max_range  = q3 + (1.5*IQR)
min_range,max_range
new_df=df[df['Coapplicant_Income']<=max_range]
new_df.shape
```

### Z-Score
```python
# Directly Applying formula
min = df['Coapplicant_Income'].mean() - (3*df['Coapplicant_Income'].std())
max = df['Coapplicant_Income'].mean() + (3*df['Coapplicant_Income'].std())
min,max
new_data = df[df['Coapplicant_Income']<= max]
z_score = (df['Coapplicant_Income'] - df['Coapplicant_Income'].mean())/(df['Coapplicant_Income'].std())
df['z_score'] = z_score
df.head()
```

