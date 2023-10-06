import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel file
data = pd.read_excel('load.xlsx')

# Assuming the columns are named 'input1', 'input2', ..., 'input6' and 'output'
input_columns = ['W', 'D', 's', 'So', 'Q','U','R/D','H','H/W','Fr','Re','shield']
output_column = 'transport'

# Extract input and output data
input_data = data[input_columns]
output_data = data[output_column]

# Create a boxplot for the output values
plt.boxplot(output_data)

# Add labels and title
plt.xlabel('Output')
plt.ylabel('Values')
plt.title('Boxplot of Output Values')

# Show the plot
plt.show()
