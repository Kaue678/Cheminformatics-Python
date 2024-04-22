import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the molecular descriptors data from the CSV file
data = pd.read_csv("molecular_descriptors.csv")

# Extract the independent variables (X) and dependent variable (y)
# Use the alias to X and Y axes: 'MolWt', 'logP', 'logD_7.4', 'HBA', 'HBD', 'NumRotBonds', 'NumAromaticRings', 'NumOxygenAtoms', 'NumNitrogenAtoms', 'TopologicalSurfaceArea', 'FractionSP3Carbons'

X = data[['logP']] # Replace 'Actual_Column_Name' with the actual column name of the target variables
y = data['TopologicalSurfaceArea']  # Replace 'Actual_Column_Name' with the actual column name of the target variables

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions using the model
predictions = model.predict(X)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions)
plt.xlabel('VAriable X')
plt.ylabel('Variable Y')
plt.title('Actual vs. Predicted values')
plt.grid(True)
plt.savefig('linear_regression_plot.png')  # Save the plot as a PNG file
plt.show()
