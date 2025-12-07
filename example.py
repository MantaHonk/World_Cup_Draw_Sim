import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# # Create a NumPy array
# array_np = np.array([1, 2, 3, 4, 5])

# # Perform operations
# mean_val = np.mean(array_np)
# sum_val = np.sum(array_np)

# # Create a Pandas DataFrame
# data = {'Name': ['Alice', 'Bob', 'Charlie'],
#         'Age': [25, 30, 22],
#         'City': ['New York', 'London', 'Paris']}
# df = pd.DataFrame(data)

# # Access columns
# names = df['Name']

# # Filter data
# young_people = df[df['Age'] < 28]

# # Create a simple line plot
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# plt.plot(x, y)
# plt.title('Sine Wave')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# # Create a scatter plot using Seaborn with a Pandas DataFrame
# sns.scatterplot(data=df, x='Age', y='Name')
# plt.title('Age vs. Name')
# plt.show()

# # Create a histogram
# sns.histplot(data=df, x='Age', kde=True)
# plt.title('Age Distribution')
# plt.show()

#heatmap
# df = pd.DataFrame([[1,0],[2,1],[3,2],[4,2],[5,0],[6,1],[7,1],[2,5],[5,3],
#                    [4,0],[1,1],[2,2],[4,3],[1,4],[2,5],[4,6],[1,7],[2,0],
#                    [2,2],[0,0],[2,1],[1,2],[1,0],[2,0],[1,1],[1,1],[1,0],
#                    [2,5],[0,0],[1,1],[1,3],[1,3],[2,2],[1,1],[0,1],[1,0]], 
#                     columns=['FTHG','FTAG'])


# df2 = pd.crosstab(df['FTHG'], df['FTAG']).div(len(df))
# sns.heatmap(df2, annot=True)
# plt.title('Soccer match heatmap')
# plt.show()

df = pd.DataFrame([['Mexico','South Africa'],['South Africa','Mexico'],['Mexico','Mexico'],['South Africa','South Africa'],
                    ['Qatar','Canada'],['Canada','Qatar'],['Qatar','Qatar'],['Canada','Canada']],
                    columns=['Team','Opponent'])


df2 = pd.crosstab(df['Team'], df['Opponent'])
print(df2)
df2.div(len(df))
sns.heatmap(df2, annot=True)
plt.title('Soccer match heatmap')
plt.show()

teams = 'teams.csv'

teams_df = pd.read_csv(teams)

# --- 1. Setup Data ---
data = {
    'Pot_ID':teams_df["Pot"].to_list(),
    'Team_ID':teams_df["Name"].to_list()
}
df_teams = pd.DataFrame(data)

# Get a list of all unique teams to define the final matrix size (A-P)
all_teams_list = sorted(df_teams['Team_ID'].unique())
num_teams = len(all_teams_list)

# Initialize an empty frequency matrix (DataFrame) with all teams as index/columns
# We use numpy for efficient counting
frequency_matrix_np = np.zeros((num_teams, num_teams), dtype=int)
team_to_index = {team: i for i, team in enumerate(all_teams_list)}


# --- 2. Function to generate one valid assignment (from previous answer) ---

def generate_one_valid_assignment(df_teams_input):
    """Generates one random assignment where each group has one team per pot."""
    pots = df_teams_input['Pot_ID'].unique()
    num_teams_per_pot = 12
    group_assignments = pd.DataFrame(columns=['Group_ID', 'Pot_ID', 'Team_ID'])
    
    for pot_id in pots:
        pot_teams = df_teams_input[df_teams_input['Pot_ID'] == pot_id]['Team_ID'].sample(frac=1, replace=False).reset_index(drop=True)
        assignments = pd.DataFrame({
            'Group_ID': range(num_teams_per_pot),
            'Pot_ID': pot_id,
            'Team_ID': pot_teams
        })
        group_assignments = pd.concat([group_assignments, assignments], ignore_index=True)
    
    return group_assignments


# --- 3. Run Many Simulations and Aggregate Frequencies ---

NUM_SIMULATIONS = 100 # More simulations = smoother, more accurate heatmap

for _ in range(NUM_SIMULATIONS):
    # Get one valid grouping
    current_assignment = generate_one_valid_assignment(df_teams)
    
    # For each created Group (Group 0, Group 1, Group 2, Group 3)
    for group_id, group_data in current_assignment.groupby('Group_ID'):
        teams_in_group = group_data['Team_ID'].tolist()
        
        # Increment frequency counts for every pairing within this specific group
        for i in range(len(teams_in_group)):
            for j in range(len(teams_in_group)):
                team_a = teams_in_group[i]
                team_b = teams_in_group[j]
                
                # We only care about unique pairs (Team A meets Team B), not A meets A
                if team_a != team_b:
                    idx_a = team_to_index[team_a]
                    idx_b = team_to_index[team_b]
                    frequency_matrix_np[idx_a, idx_b] += 1

# Convert numpy matrix back to a pandas DataFrame for plotting
frequency_matrix = pd.DataFrame(
    frequency_matrix_np,
    index=all_teams_list,
    columns=all_teams_list
)


# --- 4. Plot the Heatmap ---

# Normalize the counts to show probability/percentage instead of raw counts
# Divide by the number of simulations to get the probability of a matchup (0 to 1)
desired_order_df = df_teams.sort_values(by=['Pot_ID', 'Team_ID']).reset_index(drop=True)
sort_order = desired_order_df['Team_ID'].to_list()
probability_matrix = frequency_matrix / NUM_SIMULATIONS
probability_matrix = probability_matrix.reindex(index=sort_order, columns=sort_order)
np.fill_diagonal(probability_matrix.values, 1.0) 

plt.figure(figsize=(14, 12))
sns.heatmap(
    probability_matrix,
    annot=False,          # Keep False for 16x16 matrix
    fmt=".2f",            # Format annotations if you turn them on
    linewidths=.1,
    linecolor='white',
    cmap="YlGnBu",
    cbar_kws={'label': 'Probability of Matchup (%)'}
)

plt.title(f'Probability of Team Matchup within Valid Groups (over {NUM_SIMULATIONS} simulations)')
plt.xlabel('Team B ID')
plt.ylabel('Team A ID')
plt.show()

# Display a subset of the final probability matrix
print("\nFinal Probability Matrix (A subset of results):")
print(probability_matrix.head())