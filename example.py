import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#heatmap
df = pd.DataFrame([[1,0],[2,1],[3,2],[4,2],[5,0],[6,1],[7,1],[2,5],[5,3],
                   [4,0],[1,1],[2,2],[4,3],[1,4],[2,5],[4,6],[1,7],[2,0],
                   [2,2],[0,0],[2,1],[1,2],[1,0],[2,0],[1,1],[1,1],[1,0],
                   [2,5],[0,0],[1,1],[1,3],[1,3],[2,2],[1,1],[0,1],[1,0]], 
                    columns=['FTHG','FTAG'])


df2 = pd.crosstab(df['FTHG'], df['FTAG']).div(len(df))
sns.heatmap(df2, annot=True)
plt.title('Soccer match heatmap')
plt.show()