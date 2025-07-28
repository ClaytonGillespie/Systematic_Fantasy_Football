#%%
import pandas as pd
import pulp

# Load preprocessed players data (from previous script)
players = pd.read_parquet('./data/players_gw2.parquet')  # Adjust gameweek if needed

# Convert cost to millions
players['cost'] = players['now_cost'] / 10.0

# Position mapping
position_requirements = {
    'Goalkeeper': 2,
    'Defender': 5,
    'Midfielder': 5,
    'Forward': 3
}

TOTAL_BUDGET = 100.0
TOTAL_SQUAD_SIZE = 15
MAX_PLAYERS_PER_TEAM = 3

# Create the optimization problem
prob = pulp.LpProblem("FPL Squad Optimization", pulp.LpMaximize)

# Binary decision variables for each player
player_vars = {
    pid: pulp.LpVariable(f"player_{pid}", cat="Binary")
    for pid in players['id']
}

# Objective: Maximize total_points
prob += pulp.lpSum(players.loc[players['id'] == pid, 'total_points'].values[0] * var
                   for pid, var in player_vars.items()), "TotalPoints"

# Constraint: Total cost <= 100 million
prob += pulp.lpSum(players.loc[players['id'] == pid, 'cost'].values[0] * var
                   for pid, var in player_vars.items()) <= TOTAL_BUDGET, "TotalCost"

# Constraint: Squad size = 15
prob += pulp.lpSum(var for var in player_vars.values()) == TOTAL_SQUAD_SIZE, "SquadSize"

# Constraint: Position counts
for pos, count in position_requirements.items():
    prob += pulp.lpSum(
        var for pid, var in player_vars.items()
        if players.loc[players['id'] == pid, 'position'].values[0] == pos
    ) == count, f"{pos}_Count"

# Constraint: Max 3 players per team
for team in players['team_name'].unique():
    prob += pulp.lpSum(
        var for pid, var in player_vars.items()
        if players.loc[players['id'] == pid, 'team_name'].values[0] == team
    ) <= MAX_PLAYERS_PER_TEAM, f"TeamLimit_{team}"

# Solve the problem
print("Solving...")
prob.solve()

# Display results
selected = [pid for pid, var in player_vars.items() if var.value() == 1]
squad = players[players['id'].isin(selected)]

selected
#%%