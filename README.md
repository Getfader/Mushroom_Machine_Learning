# Mushroom Classification Project

## Overview

This project explores the classification of mushrooms using a dataset from the UCI Machine Learning Repository, specifically the Secondary Mushroom Dataset. The primary goal was to practice tuning machine learning models. However, the dataset turned out to be extremely well-sorted, leading to perfect classification without significant tuning. Despite this unexpected outcome, the project serves as a valuable exploration into the nuances of dataset preparation and the importance of challenging datasets in machine learning.

## Dataset

Source: Secondary Mushroom Dataset
Description: The dataset consists of 61,069 entries and 21 columns, including features such as cap-diameter, cap-shape, gill-color, habitat, and others. The target variable is class, indicating whether a mushroom is poisonous (p) or edible (e).
## Installation

To run this project, you'll need Python and the following packages:

pandas
numpy
scikit-learn
matplotlib
Install the dependencies with:

```sh
pip install pandas numpy scikit-learn matplotlib
```

## Data Exploration

The dataset was loaded and explored using pandas. Key steps included:

### Data Loading:

```python
import pandas as pd
df = pd.read_csv('secondary_data.csv', delimiter=';')
```

### Data Overview

Displaying information and basic statistics of the dataset:

```python
df.info()
df.describe()
```

### Categorical Features Analysis

Counting unique categories in nominal columns:

```python
nominal_columns = ['class', 'cap-shape', 'cap-surface', 'cap-color',
'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root',
'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type',
'spore-print-color', 'habitat', 'season']

total_categories = sum(df[col].nunique() for col in nominal_columns)
print("Total number of categories in all nominal columns:", total_categories)
```

## Challenges and Outcomes

The dataset's sorting quality resulted in a perfectly tuned model without any additional adjustments. While this was not ideal for tuning practice, it underscored the importance of using more complex and less obvious datasets for model training and tuning exercises.

## Future Work

Future work involves:

Dataset Selection: Choosing datasets with greater complexity or noise to better understand the impact of hyperparameter tuning.
Dimensionality Reduction: Applying PCA or other techniques to investigate feature groupings and data relationships.
Visualization: Plotting height, diameter, and width against the target variable to explore potential patterns.
## Running the Code

### Data Preprocessing

One-hot encode nominal variables:

```python
df_encoded = pd.get_dummies(df, columns=nominal_columns).astype(float)
```

### Model Training

Train a classifier (e.g., decision tree, random forest) on the encoded data and evaluate its performance.

### Analysis

Analyze model results and discuss potential improvements or observations.

## Conclusion

This project serves as a reminder of the necessity of challenging datasets in machine learning for effective model tuning and validation. The project will remain available on GitHub for reference and as a stepping stone for further exploration with different datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
