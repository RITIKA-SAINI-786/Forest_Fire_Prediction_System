import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv(".venv\Forest_fire.csv")

# Split the data into features (X) and target variable (y)
X = data.iloc[:, 1:-1]  # Assuming features are from column 1 to second-last
y = data.iloc[:, -1]    # Assuming target is in the last column

# Convert to numpy arrays and ensure integer type
X = X.astype('int')
y = y.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Instantiate and train the Random Forest model
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)

# Check accuracy on the test set
accuracy = random_forest.score(X_test, y_test)
print(f"Accuracy of Random Forest model: {accuracy:.2f}")

# Serialize the trained model to a file
pickle.dump(random_forest, open('model.pkl', 'wb'))
