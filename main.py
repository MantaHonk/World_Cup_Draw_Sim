import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from country import Country

#RESTRICTIONS
"""
No two of the same conference (except UEFA)
Teams ranked 1, 2, then 3, 4 must be in opposite brackets
ABCJKL,DEFGHI
Mex A1
Can B1
USA D1
Each group requires 1 UEFA
"""

teams = 'teams.csv'

teams_df = pd.read_csv(teams)
print(teams_df.head())

data = {
    'Pot_ID':teams_df["Pot"].to_list(),
    'Team_ID':teams_df["Name"].to_list(),
    'Ranking':teams_df["Ranking"].to_list(),
    'Confederation':teams_df["Confederation"].to_list()
}
df_teams = pd.DataFrame(data)

all_teams_list = df_teams['Team_ID'].unique()
num_teams = len(all_teams_list)

# Initialize an empty frequency matrix with all teams as index/columns
frequency_matrix_np = np.zeros((num_teams, num_teams), dtype=int)
team_to_index = {team: i for i, team in enumerate(all_teams_list)}


def generate_one_valid_assignment(df_teams_input):
    """Generates one random assignment where each group has one team per pot."""
    pots = df_teams_input['Pot_ID'].unique()
    teams = df_teams_input['Team_ID'].unique()
    groups = len(teams)//len(pots)#teams.div(pots)
    group_assignments = pd.DataFrame(columns=['Group_ID', 'Pot_ID', 'Team_ID'])
    
    for pot_id in pots:
        
        pot_teams = df_teams_input[df_teams_input['Pot_ID'] == pot_id]
        sample_order = pot_teams['Team_ID'].sample(n = groups, replace=False).reset_index(drop=True) 
        assignments = pd.DataFrame({
            'Group_ID': range(1,1+groups),
            'Pot_ID': pot_id,
            'Team_ID': sample_order
        })
        group_assignments = pd.concat([group_assignments, assignments], ignore_index=True)
        print(group_assignments)
    return group_assignments



NUM_SIMULATIONS = 1 # More simulations = smoother, more accurate heatmap

for _ in range(NUM_SIMULATIONS):
    current_assignment = generate_one_valid_assignment(df_teams)
    
    for group_id, group_data in current_assignment.groupby('Group_ID'):
        teams_in_group = group_data['Team_ID'].tolist()
        
        for i in range(len(teams_in_group)):
            for j in range(len(teams_in_group)):
                team_a = teams_in_group[i]
                team_b = teams_in_group[j]
                    
                idx_a = team_to_index[team_a]
                idx_b = team_to_index[team_b]
                frequency_matrix_np[idx_a, idx_b] += 1

# Convert numpy matrix back to a pandas DataFrame for plotting
frequency_matrix = pd.DataFrame(
    frequency_matrix_np,
    index=all_teams_list,
    columns=all_teams_list
)



# Normalize the counts to show probability/percentage instead of raw counts
# Divide by the number of simulations to get the probability of a matchup (0 to 1) 
probability_matrix = frequency_matrix / NUM_SIMULATIONS

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    probability_matrix*100,
    annot=True,
    ax = ax,
    annot_kws={"fontsize": 6},          
    fmt=".0f",            # Format annotations if you turn them on
    # linewidths=.1,
    # linecolor='white',
    cmap="bone_r", #"YlGnBu"
    cbar_kws={'label': 'Probability of Matchup (%)','shrink': 0.6},
    square = True,
    vmax = 30
)

plt.title(f'Probability of Team Matchup within Valid Groups (over {NUM_SIMULATIONS} simulations)')
plt.xlabel('Team B ID', fontsize = 8)
plt.ylabel('Team A ID', fontsize = 8)

ax.tick_params(axis='x', labelsize=6) 
ax.tick_params(axis='y', labelsize=6) 
plt.show()

print("\nFinal Probability Matrix:")
print(probability_matrix.head())