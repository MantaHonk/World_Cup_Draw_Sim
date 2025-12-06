import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from country import Country
# Create a NumPy array
array_np = np.array([1, 2, 3, 4, 5])

# Perform operations
mean_val = np.mean(array_np)
sum_val = np.sum(array_np)

# Create a Pandas DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

# Access columns
names = df['Name']

# Filter data
young_people = df[df['Age'] < 28]

# Create a simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# Create a scatter plot using Seaborn with a Pandas DataFrame
sns.scatterplot(data=df, x='Age', y='Name')
plt.title('Age vs. Name')
plt.show()

# Create a histogram
sns.histplot(data=df, x='Age', kde=True)
plt.title('Age Distribution')
plt.show()