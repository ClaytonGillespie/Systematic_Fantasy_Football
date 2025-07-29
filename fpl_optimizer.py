#%%
import pandas as pd
import pulp

def optimize_fpl_squad(player_data_path, n_playing=11, budget=100.0):
    players = pd.read_parquet(player_data_path)

    players['cost'] = players['now_cost'] / 10.0

    position_requirements = {
        'Goalkeeper': 2,
        'Defender': 5,
        'Midfielder': 5,
        'Forward': 3
    }

    total_squad_size = 15
    max_per_team = 3

    # Create LP model
    prob = pulp.LpProblem("FPL Squad Optimization", pulp.LpMaximize)

    # Variables
    player_vars = {pid: pulp.LpVariable(f"select_{pid}", cat="Binary")
                   for pid in players['id']}
    playing_vars = {pid: pulp.LpVariable(f"playing_{pid}", cat="Binary")
                    for pid in players['id']}
    captain_vars = {pid: pulp.LpVariable(f"captain_{pid}", cat="Binary")
                    for pid in players['id']}

    # Objective: playing points + captain bonus (captain counts again)
    prob += pulp.lpSum(
        (players.loc[players['id'] == pid, 'total_points'].values[0] *
         (playing_vars[pid] + captain_vars[pid]))
        for pid in players['id']
    ), "TotalPointsWithCaptain"

    # Total cost
    prob += pulp.lpSum(players.loc[players['id'] == pid, 'cost'].values[0] * player_vars[pid]
                       for pid in players['id']) <= budget, "Budget"

    # Squad size = 15
    prob += pulp.lpSum(player_vars[pid] for pid in players['id']) == total_squad_size, "SquadSize"

    # Position constraints
    for pos, required in position_requirements.items():
        prob += pulp.lpSum(player_vars[pid]
                           for pid in players['id']
                           if players.loc[players['id'] == pid, 'position'].values[0] == pos) == required, f"Position_{pos}"

    # Team constraint: max 3 per team
    for team in players['team_name'].unique():
        prob += pulp.lpSum(player_vars[pid]
                           for pid in players['id']
                           if players.loc[players['id'] == pid, 'team_name'].values[0] == team) <= max_per_team, f"TeamLimit_{team}"

    # Playing constraint
    prob += pulp.lpSum(playing_vars[pid] for pid in players['id']) == n_playing, "NumPlaying"

    # Minimum formation rules
    minimum_playing = {
        'Goalkeeper': 1,
        'Defender': 3,
        'Midfielder': 2,
        'Forward': 1
    }

    for pos, min_required in minimum_playing.items():
        prob += pulp.lpSum(
            playing_vars[pid]
            for pid in players['id']
            if players.loc[players['id'] == pid, 'position'].values[0] == pos
        ) >= min_required, f"MinPlaying_{pos}"

    # Captain constraint
    prob += pulp.lpSum(captain_vars[pid] for pid in players['id']) == 1, "OneCaptain"

    # Linking: captain must be playing
    for pid in players['id']:
        prob += captain_vars[pid] <= playing_vars[pid], f"CaptainPlays_{pid}"
        prob += playing_vars[pid] <= player_vars[pid], f"PlayOnlyIfSelected_{pid}"

    # Solve
    print("Solving...")
    prob.solve()
    print(f"✅ Solved: {pulp.LpStatus[prob.status]}")

    # Extract solution
    selected_ids = [pid for pid in players['id'] if player_vars[pid].value() == 1]
    playing_ids = [pid for pid in players['id'] if playing_vars[pid].value() == 1]
    captain_id = [pid for pid in players['id'] if captain_vars[pid].value() == 1][0]

    squad = players[players['id'].isin(selected_ids)].copy()
    squad['is_playing'] = squad['id'].isin(playing_ids)
    squad['is_captain'] = squad['id'] == captain_id

    # Show results
    print("\n📝 Final Squad:")
    print(squad[['first_name', 'second_name', 'position', 'team_name',
                 'cost', 'total_points', 'is_playing', 'is_captain']])

    print(f"\n💰 Total cost: £{squad['cost'].sum():.1f}m")
    playing_points = squad[squad['is_playing']]['total_points'].sum()
    captain_bonus = squad.loc[squad['is_captain'], 'total_points'].values[0]
    print(f"⭐ Total points (with captain bonus): {playing_points + captain_bonus}")

    return squad

#%%
squad = optimize_fpl_squad('./data/players_gw0.parquet', n_playing=11)
squad
#%%