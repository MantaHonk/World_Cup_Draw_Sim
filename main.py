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
#print(teams_df.head())

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
    groups = len(teams)//len(pots)
    group_assignments = pd.DataFrame(columns=['Group_ID', 'Pot_ID', 'Team_ID', 'Confederation'])
    IC_PATH_A_POTENTIAL_CONFS = ["OFC", "CONCACAF", "CAF"] 
    IC_PATH_B_POTENTIAL_CONFS = ["AFC", "CONMEBOL", "CONCACAF"]
    
    for pot_id in pots:
        
        pot_teams = df_teams_input[df_teams_input['Pot_ID'] == pot_id]
        
        for g in range(1,1+groups):
            
            group = group_assignments[group_assignments['Group_ID'] == g]
            conf_in_group = group['Confederation']
            
            valid_teams = pot_teams[~pot_teams['Confederation'].isin(conf_in_group)]
            if (conf_in_group == "UEFA").sum() < 2:
                valid_teams = pd.concat([valid_teams, pot_teams[pot_teams['Confederation'] == "UEFA"]])
            
            if "IC Path A" in valid_teams["Team_ID"].values:
                conflicts_with_ic_path_a = any(conf in conf_in_group.to_list() for conf in IC_PATH_A_POTENTIAL_CONFS)
                if conflicts_with_ic_path_a:
                    #print("Removing IC Path A due to conflict")
                    valid_teams = valid_teams[valid_teams['Team_ID'] != "IC Path A"]
            if "IC Path B" in valid_teams["Team_ID"].values:
                conflicts_with_ic_path_b = any(conf in conf_in_group.to_list() for conf in IC_PATH_B_POTENTIAL_CONFS)
                if conflicts_with_ic_path_b:
                    #print("Removing IC Path B due to conflict")
                    valid_teams = valid_teams[valid_teams['Team_ID'] != "IC Path B"]
            #print(group)
            #print(valid_teams)
            
            if valid_teams.empty:
                #print(f"Could not find a valid team for Group {g} from Pot {pot_id}. Draw failed.")
                raise ValueError("Draw failed due to constraints.")
            else:
                team_drawn_row = valid_teams.sample(n=1, replace=False)
            
            new_assignment = {
                'Group_ID': g,
                'Pot_ID': pot_id,
                'Team_ID': team_drawn_row['Team_ID'].iloc[0],
                'Confederation': team_drawn_row['Confederation'].iloc[0]
            }

            group_assignments = pd.concat(
                [group_assignments, pd.DataFrame([new_assignment])], 
                ignore_index=True
            )

            pot_teams = pot_teams.drop(team_drawn_row.index)

        #print(group_assignments)
    return group_assignments



NUM_SIMULATIONS = 1000 # More simulations = smoother, more accurate heatmap
valid_sims = 0
num_invalid = 0
num_cycles = 0
while valid_sims < NUM_SIMULATIONS:
    num_cycles += 1
    try:
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
        valid_sims += 1
        if valid_sims % 100 == 0:
            print(f"Completed {valid_sims} valid simulations so far")
    except ValueError:
        num_invalid += 1
        continue
print(f"Completed {valid_sims} valid simulations with {num_invalid} invalid attempts over {num_cycles} total cycles.")
# Convert numpy matrix back to a pandas DataFrame for plotting
frequency_matrix = pd.DataFrame(
    frequency_matrix_np,
    index=all_teams_list,
    columns=all_teams_list
)



# Normalize the counts to show probability/percentage instead of raw counts
probability_matrix = frequency_matrix / NUM_SIMULATIONS

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    probability_matrix*100,
    annot=True,
    ax = ax,
    annot_kws={"fontsize": 6},          
    fmt=".0f",
    # linewidths=.1,
    # linecolor='white',
    cmap="bone_r", #"YlGnBu"
    cbar_kws={'label': 'Probability of Matchup (%)','shrink': 0.6},
    square = True,
    vmax = 30
)

plt.title(f'Probability of Team Matchup For 2026 FIFA World Cup (over {NUM_SIMULATIONS} simulations)')

ax.tick_params(axis='x', labelsize=6) 
ax.tick_params(axis='y', labelsize=6) 
plt.show()

print("\nFinal Probability Matrix:")
print(probability_matrix.head())