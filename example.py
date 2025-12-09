import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# df = pd.DataFrame([['Mexico','South Africa'],['South Africa','Mexico'],['Mexico','Mexico'],['South Africa','South Africa'],
#                     ['Qatar','Canada'],['Canada','Qatar'],['Qatar','Qatar'],['Canada','Canada']],
#                     columns=['Team','Opponent'])


# df2 = pd.crosstab(df['Team'], df['Opponent'])
# print(df2)
# df2.div(len(df))
# sns.heatmap(df2, annot=True)
# plt.title('Soccer match heatmap')
# plt.show()

teams = 'teams.csv'

teams_df = pd.read_csv(teams)

# --- 1. Setup Data ---
data = {
    'Pot_ID':teams_df["Pot"].to_list(),
    'Team_ID':teams_df["Name"].to_list(),
    'Ranking':teams_df["Ranking"].to_list(),
    'Confederation':teams_df["Confederation"].to_list()
}
df_teams = pd.DataFrame(data)

# Get a list of all unique teams to define the final matrix size (A-P)
all_teams_list = df_teams['Team_ID'].unique()
num_teams = len(all_teams_list)

# Initialize an empty frequency matrix (DataFrame) with all teams as index/columns
# We use numpy for efficient counting
frequency_matrix_np = np.zeros((num_teams, num_teams), dtype=int)
team_to_index = {team: i for i, team in enumerate(all_teams_list)}


# --- 2. Function to generate one valid assignment (from previous answer) ---


def generate_one_valid_assignment(df_teams_input):
    """Generates one random assignment where each group has one team per pot."""
    
    # 1. Setup metadata
    pots = df_teams_input['Pot_ID'].unique()
    teams = df_teams_input['Team_ID'].unique()
    # Calculate the number of groups (e.g., 32 teams / 4 pots = 8 groups)
    groups = len(teams) // len(pots) 
    
    # Initialize the master DataFrame where we record all assignments as we draw them
    # This DF starts empty but mirrors the structure of df_teams_input
    master_assignments = pd.DataFrame(columns=['Group_ID', 'Pot_ID', 'Team_ID', 'Confederation'])
    
    # 2. Iterate through each pot one by one
    for pot_id in pots:
        # Start with a fresh pool of teams *available* in this specific pot
        pot_teams_pool = df_teams_input[df_teams_input['Pot_ID'] == pot_id].copy()
        
        # 3. Iterate through each group (G1 to G8, for example) to place a team from the current pot
        for g in range(1, 1 + groups):
            
            # Assuming this is inside your main loops:

            # ... (inside the loop for g in range...)

            group = master_assignments[master_assignments['Group_ID'] == g]
            conf_in_group = group['Confederation'].unique()

            EUROPE_CONFED = 'UEFA' # Make sure this matches your actual DataFrame values
            MAX_EURO_PER_GROUP = 2

            # 1. Start with a strict filter: Exclude any confederation already present
            strict_conflicts = [conf for conf in conf_in_group if conf != EUROPE_CONFED]
            valid_teams = pot_teams_pool[~pot_teams_pool['Confederation'].isin(conf_in_group != "UEFA")]

            # 2. Add the UEFA-specific condition:
            euro_count_in_group = (group['Confederation'] == EUROPE_CONFED).sum()

            if euro_count_in_group >= MAX_EURO_PER_GROUP:
                # If the group already has 2 or more Euro teams, we must filter out any remaining Euro teams from valid_teams
                valid_teams = valid_teams[valid_teams['Confederation'] != EUROPE_CONFED]
                
            # At this point, 'valid_teams' contains all teams that meet all constraints.


# ... rest of your code to sample from valid_teams and update assignments ...
            
            if valid_teams.empty:
                # This should ideally not happen if constraints allow a full draw
                print(f"Could not find a valid team for Group {g} from Pot {pot_id}. Draw failed.")
                team_drawn_row = pot_teams_pool.sample(n=1,replace=False)
            else:
            # C. Draw one random team from the valid candidates
            # We sample the entire row to get both Team_ID and Confederation
                team_drawn_row = valid_teams.sample(n=1, replace=False)
            
            # D. Record the assignment in the master list
            new_assignment = {
                'Group_ID': g,
                'Pot_ID': pot_id,
                'Team_ID': team_drawn_row['Team_ID'].iloc[0],
                'Confederation': team_drawn_row['Confederation'].iloc[0]
            }
            # Use pd.concat to append the new assignment row efficiently
            master_assignments = pd.concat(
                [master_assignments, pd.DataFrame([new_assignment])], 
                ignore_index=True
            )
            
            # E. Remove the drawn team from the current pot's pool so it can't be selected again
            pot_teams_pool = pot_teams_pool.drop(team_drawn_row.index)
            # The 'pot_teams_pool' variable is now smaller for the next iteration of 'g'

    # The function returns the complete list of all valid assignments
    # Sort for cleaner output if desired: master_assignments.sort_values(by=['Group_ID', 'Pot_ID'])
    return master_assignments



# --- 3. Run Many Simulations and Aggregate Frequencies ---

NUM_SIMULATIONS = 1 # More simulations = smoother, more accurate heatmap

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
                    
                idx_a = team_to_index[team_a]
                idx_b = team_to_index[team_b]
                frequency_matrix_np[idx_a, idx_b] += 1
                # We only care about unique pairs (Team A meets Team B), not A meets A
                # if team_a != team_b:
                #     idx_a = team_to_index[team_a]
                #     idx_b = team_to_index[team_b]
                #     frequency_matrix_np[idx_a, idx_b] += 1

# Convert numpy matrix back to a pandas DataFrame for plotting
frequency_matrix = pd.DataFrame(
    frequency_matrix_np,
    index=all_teams_list,
    columns=all_teams_list
)


# --- 4. Plot the Heatmap ---

# Normalize the counts to show probability/percentage instead of raw counts
# Divide by the number of simulations to get the probability of a matchup (0 to 1)
# desired_order_df = df_teams.sort_values(by=['Pot_ID', 'Team_ID']).reset_index(drop=True)
# sort_order = desired_order_df['Team_ID'].to_list()
probability_matrix = frequency_matrix / NUM_SIMULATIONS
# probability_matrix = probability_matrix.reindex(index=sort_order, columns=sort_order)
#np.fill_diagonal(probability_matrix.values, 1.0) 

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

ax.tick_params(axis='x', labelsize=6)  # Set X-axis label font size (e.g., 8)
ax.tick_params(axis='y', labelsize=6) 
plt.show()

# Display a subset of the final probability matrix
print("\nFinal Probability Matrix (A subset of results):")
print(probability_matrix.head())