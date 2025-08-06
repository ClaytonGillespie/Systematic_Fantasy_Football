#%%
import pandas as pd
import pulp

def try_multiple_solvers(prob, verbose=True):
    """
    Try multiple PuLP solvers and return the best result
    
    Args:
        prob: PuLP problem instance
        verbose: Whether to show solver progress
    
    Returns:
        tuple: (best_status, best_solver_name, objective_value)
    """
    # Define available solvers to try
    solvers_to_try = [
        ('PULP_CBC_CMD', pulp.PULP_CBC_CMD(msg=0)),
        ('CPLEX_CMD', pulp.CPLEX_CMD(msg=0)),
        ('GUROBI_CMD', pulp.GUROBI_CMD(msg=0)),
        ('SCIP_CMD', pulp.SCIP_CMD(msg=0)),
        ('GLPK_CMD', pulp.GLPK_CMD(msg=0)),
        ('COIN_CMD', pulp.COIN_CMD(msg=0))
    ]
    
    results = []
    best_status = None
    best_solver = None
    best_objective = float('-inf')
    
    if verbose:
        print("🔄 Trying multiple solvers...")
    
    for solver_name, solver in solvers_to_try:
        try:
            # Create a copy of the problem for each solver
            prob_copy = prob.copy()
            
            if verbose:
                print(f"   Testing {solver_name}...", end="")
            
            # Try to solve with this solver
            prob_copy.solve(solver)
            status = pulp.LpStatus[prob_copy.status]
            
            if status == 'Optimal':
                objective_value = pulp.value(prob_copy.objective)
                results.append({
                    'solver': solver_name,
                    'status': status,
                    'objective': objective_value,
                    'time': 'N/A'  # PuLP doesn't provide timing info easily
                })
                
                if objective_value > best_objective:
                    best_objective = objective_value
                    best_solver = solver_name
                    best_status = status
                    # Copy the solution back to original problem
                    for var in prob.variables():
                        var.varValue = prob_copy.variablesDict()[var.name].varValue
                    prob.status = prob_copy.status
                
                if verbose:
                    print(f" ✅ Optimal (Objective: {objective_value:.2f})")
            else:
                results.append({
                    'solver': solver_name,
                    'status': status,
                    'objective': None,
                    'time': 'N/A'
                })
                if verbose:
                    print(f" ❌ {status}")
        
        except Exception as e:
            if verbose:
                print(f" ⚠️ Error: {str(e)[:50]}...")
            results.append({
                'solver': solver_name,
                'status': 'Error',
                'objective': None,
                'time': 'N/A'
            })
            continue
    
    # Display results summary
    if verbose and results:
        print(f"\n📊 SOLVER COMPARISON:")
        print("-" * 60)
        print(f"{'Solver':<15} {'Status':<12} {'Objective':<12} {'Notes'}")
        print("-" * 60)
        
        for result in results:
            obj_str = f"{result['objective']:.2f}" if result['objective'] else "N/A"
            notes = "🏆 Best" if result['solver'] == best_solver else ""
            print(f"{result['solver']:<15} {result['status']:<12} {obj_str:<12} {notes}")
        
        if best_solver:
            print(f"\n🎯 Best solver: {best_solver} (Objective: {best_objective:.2f})")
        else:
            print(f"\n⚠️ No solver found optimal solution")
    
    return best_status, best_solver, best_objective

def optimize_fpl_squad(expected_points_path, gameweek=0, n_playing=11, budget=100.0, exclude_injured=True, 
                      bench_gk_weight=0.5, bench_outfield_weight=0.3, min_bench_outfield_points=2.0,
                      try_multiple_solvers_flag=True, solver_verbose=True):
    """
    Optimize FPL squad using expected points predictions
    
    Args:
        expected_points_path: Path to the expected points CSV
        gameweek: Gameweek number for file naming
        n_playing: Number of playing players (11)
        budget: Budget constraint in millions
        exclude_injured: Whether to exclude injured/doubtful players
        bench_gk_weight: Weight for bench goalkeeper points (0.5 = half value)
        bench_outfield_weight: Weight for bench outfield players (0.3 = 30% value)
        min_bench_outfield_points: Minimum expected points for best bench outfield player
        try_multiple_solvers_flag: Whether to try multiple solvers for best result
        solver_verbose: Whether to show solver comparison details
    """
    
    # Load expected points data
    if expected_points_path.endswith('.csv'):
        players = pd.read_csv(expected_points_path)
    else:
        # If path doesn't include extension, add gameweek-specific naming
        players = pd.read_csv(f'./data_2025/player_expected_scores_gw{gameweek}.csv')
    
    print(f"Loaded {len(players)} players from expected points data")
    
    # Convert cost to millions if needed
    if 'cost_per_million' in players.columns:
        players['cost'] = players['cost_per_million']
    else:
        players['cost'] = players['now_cost'] / 10.0
    
    # Use expected points as the optimization target
    players['points_to_optimize'] = players['expected_points_next_4gw']
    
    # Filter out injured players if requested
    if exclude_injured and 'injury_status' in players.columns:
        original_count = len(players)
        players = players[players['injury_status'] == 'available'].copy()
        excluded_count = original_count - len(players)
        if excluded_count > 0:
            print(f"🚑 Excluded {excluded_count} injured/doubtful players")
    
    # Handle any missing values
    players = players.dropna(subset=['points_to_optimize', 'cost'])
    
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
                   for pid in players['element_id']}
    playing_vars = {pid: pulp.LpVariable(f"playing_{pid}", cat="Binary")
                    for pid in players['element_id']}
    captain_vars = {pid: pulp.LpVariable(f"captain_{pid}", cat="Binary")
                    for pid in players['element_id']}
    bench_gk_var = pulp.LpVariable("bench_gk_selected", cat="Binary")
    bench_outfield_vars = {pid: pulp.LpVariable(f"bench_outfield_{pid}", cat="Binary")
                          for pid in players['element_id'] if players.loc[players['element_id'] == pid, 'element_type'].values[0] != 'Goalkeeper'}

    # Objective: playing points + captain bonus + weighted bench value
    objective = pulp.lpSum(
        (players.loc[players['element_id'] == pid, 'points_to_optimize'].values[0] *
         (playing_vars[pid] + captain_vars[pid]))
        for pid in players['element_id']
    )
    
    # Add bench goalkeeper value (weighted)
    objective += pulp.lpSum(
        (players.loc[players['element_id'] == pid, 'points_to_optimize'].values[0] * 
         bench_gk_weight * (player_vars[pid] - playing_vars[pid]))
        for pid in players['element_id']
        if players.loc[players['element_id'] == pid, 'element_type'].values[0] == 'Goalkeeper'
    )
    
    # Add bench outfield value (weighted)
    objective += pulp.lpSum(
        (players.loc[players['element_id'] == pid, 'points_to_optimize'].values[0] * 
         bench_outfield_weight * (player_vars[pid] - playing_vars[pid]))
        for pid in players['element_id']
        if players.loc[players['element_id'] == pid, 'element_type'].values[0] != 'Goalkeeper'
    )
    
    prob += objective, "TotalExpectedPointsWithBenchValue"

    # Total cost constraint
    prob += pulp.lpSum(players.loc[players['element_id'] == pid, 'cost'].values[0] * player_vars[pid]
                       for pid in players['element_id']) <= budget, "Budget"

    # Squad size = 15
    prob += pulp.lpSum(player_vars[pid] for pid in players['element_id']) == total_squad_size, "SquadSize"

    # Position constraints
    for pos, required in position_requirements.items():
        prob += pulp.lpSum(player_vars[pid]
                           for pid in players['element_id']
                           if players.loc[players['element_id'] == pid, 'element_type'].values[0] == pos) == required, f"Position_{pos}"

    # Team constraint: max 3 per team
    for team in players['team'].unique():
        prob += pulp.lpSum(player_vars[pid]
                           for pid in players['element_id']
                           if players.loc[players['element_id'] == pid, 'team'].values[0] == team) <= max_per_team, f"TeamLimit_{team}"

    # Playing constraint
    prob += pulp.lpSum(playing_vars[pid] for pid in players['element_id']) == n_playing, "NumPlaying"

    # Minimum formation rules (3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1)
    minimum_playing = {
        'Goalkeeper': 1,
        'Defender': 3,  # Minimum 3 defenders
        'Midfielder': 2, # Minimum 2 midfielders  
        'Forward': 1     # Minimum 1 forward
    }

    for pos, min_required in minimum_playing.items():
        prob += pulp.lpSum(
            playing_vars[pid]
            for pid in players['element_id']
            if players.loc[players['element_id'] == pid, 'element_type'].values[0] == pos
        ) >= min_required, f"MinPlaying_{pos}"
    
    # Maximum playing constraints to prevent invalid formations
    maximum_playing = {
        'Goalkeeper': 1,  # Exactly 1 GK
        'Defender': 5,    # Max 5 defenders
        'Midfielder': 5,  # Max 5 midfielders
        'Forward': 3      # Max 3 forwards
    }
    
    for pos, max_allowed in maximum_playing.items():
        prob += pulp.lpSum(
            playing_vars[pid]
            for pid in players['element_id']
            if players.loc[players['element_id'] == pid, 'element_type'].values[0] == pos
        ) <= max_allowed, f"MaxPlaying_{pos}"

    # Captain constraint: exactly one captain
    prob += pulp.lpSum(captain_vars[pid] for pid in players['element_id']) == 1, "OneCaptain"

    # Linking constraints
    for pid in players['element_id']:
        # Captain must be playing
        prob += captain_vars[pid] <= playing_vars[pid], f"CaptainPlays_{pid}"
        # Can only play if selected in squad
        prob += playing_vars[pid] <= player_vars[pid], f"PlayOnlyIfSelected_{pid}"
    
    # Bench quality constraints
    # Ensure at least one decent bench outfield player
    prob += pulp.lpSum(
        (player_vars[pid] - playing_vars[pid]) * players.loc[players['element_id'] == pid, 'points_to_optimize'].values[0]
        for pid in players['element_id']
        if players.loc[players['element_id'] == pid, 'element_type'].values[0] != 'Goalkeeper'
    ) >= min_bench_outfield_points, "MinBenchOutfieldQuality"

    # Solve the optimization with multiple solvers
    if try_multiple_solvers_flag:
        status, best_solver, objective_value = try_multiple_solvers(prob, verbose=solver_verbose)
        if status == 'Optimal':
            print(f"✅ Optimization completed with {best_solver}")
        else:
            print(f"⚠️ Best result: {status} with {best_solver}")
    else:
        # Use default CBC solver
        print("Solving with CBC solver...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        best_solver = "PULP_CBC_CMD"
        print(f"✅ Optimization status: {status}")
    
    if status not in ['Optimal', 'Feasible']:
        print("⚠️ Warning: Could not find optimal solution")
        return None

    # Extract solution
    selected_ids = [pid for pid in players['element_id'] if player_vars[pid].value() == 1]
    playing_ids = [pid for pid in players['element_id'] if playing_vars[pid].value() == 1]
    captain_id = [pid for pid in players['element_id'] if captain_vars[pid].value() == 1][0]

    # Create final squad dataframe
    squad = players[players['element_id'].isin(selected_ids)].copy()
    squad['is_playing'] = squad['element_id'].isin(playing_ids)
    squad['is_captain'] = squad['element_id'] == captain_id
    
    # Sort by position and expected points
    position_order = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
    squad['position_order'] = squad['element_type'].map(position_order)
    squad = squad.sort_values(['position_order', 'points_to_optimize'], ascending=[True, False])

    # Display results
    print("\n📝 OPTIMIZED SQUAD (Next 4 Gameweeks):")
    print("=" * 80)
    
    display_cols = ['element_id', 'web_name', 'element_type', 'team', 'cost', 'points_to_optimize', 'is_playing', 'is_captain']
    if 'value_score' in squad.columns:
        display_cols.insert(-2, 'value_score')
    if 'injury_status' in squad.columns:
        display_cols.insert(-2, 'injury_status')
    
    # Show playing XI first
    playing_squad = squad[squad['is_playing']].copy()
    bench = squad[~squad['is_playing']].copy()
    
    print("\n⭐ STARTING XI:")
    print(playing_squad[display_cols].to_string(index=False))
    
    print("\n🪑 BENCH:")
    bench_sorted = bench.sort_values('points_to_optimize', ascending=False)
    print(bench_sorted[display_cols].to_string(index=False))
    
    # Bench analysis
    bench_gk = bench[bench['element_type'] == 'Goalkeeper']
    bench_outfield = bench[bench['element_type'] != 'Goalkeeper']
    
    print(f"\n🔄 BENCH ANALYSIS:")
    if len(bench_gk) > 0:
        gk_points = bench_gk.iloc[0]['points_to_optimize']
        gk_id = bench_gk.iloc[0]['element_id']
        print(f"  Bench GK: {bench_gk.iloc[0]['web_name']} (ID: {gk_id}) - {gk_points:.1f} expected points")
    
    if len(bench_outfield) > 0:
        best_bench = bench_outfield.iloc[0]
        print(f"  Best bench outfield: {best_bench['web_name']} (ID: {best_bench['element_id']}) - {best_bench['points_to_optimize']:.1f} expected points")
        bench_total = bench_outfield['points_to_optimize'].sum()
        print(f"  Total bench outfield value: {bench_total:.1f} expected points")
    
    # Summary statistics
    total_cost = squad['cost'].sum()
    playing_points = playing_squad['points_to_optimize'].sum()
    captain_bonus = squad.loc[squad['is_captain'], 'points_to_optimize'].values[0]
    total_expected = playing_points + captain_bonus
    
    print(f"\n📊 SQUAD SUMMARY:")
    print(f"💰 Total cost: £{total_cost:.1f}m (Budget remaining: £{budget - total_cost:.1f}m)")
    print(f"⭐ Expected points (4GW): {total_expected:.1f} (Playing: {playing_points:.1f} + Captain: {captain_bonus:.1f})")
    print(f"📈 Points per million: {total_expected / total_cost:.2f}")
    print(f"🔧 Solved with: {best_solver}")
    
    # Formation analysis
    formation = {}
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        formation[pos] = len(playing_squad[playing_squad['element_type'] == pos])
    
    formation_str = f"{formation['Goalkeeper']}-{formation['Defender']}-{formation['Midfielder']}-{formation['Forward']}"
    print(f"🏟️  Formation: {formation_str}")
    
    # Team distribution
    print(f"\n🏆 TEAM DISTRIBUTION:")
    team_counts = squad['team'].value_counts()
    for team, count in team_counts.items():
        print(f"  {team}: {count} players")
    
    # Show captain with ID
    captain_info = squad[squad['is_captain']].iloc[0]
    print(f"\n👑 Captain: {captain_info['web_name']} (ID: {captain_info['element_id']}) ({captain_info['element_type']}) - {captain_info['points_to_optimize']:.1f} expected points")
    
    return squad

#%%
# Usage examples with multiple solvers:

# Option 1: Use multiple solvers (recommended)
def optimize_with_expected_points(gameweek=0, budget=100.0):
    """Optimize squad using generated expected points with multiple solvers"""
    csv_path = f'./data_2025/player_expected_scores_gw{gameweek}.csv'
    return optimize_fpl_squad(csv_path, gameweek=gameweek, budget=budget, exclude_injured=True,
                            bench_gk_weight=0.75, bench_outfield_weight=0.2, min_bench_outfield_points=3.0,
                            try_multiple_solvers_flag=True, solver_verbose=True)

# Option 2: Single solver (faster)
def optimize_single_solver(gameweek=0, budget=100.0):
    """Optimize with single CBC solver for speed"""
    csv_path = f'./data_2025/player_expected_scores_gw{gameweek}.csv'
    return optimize_fpl_squad(csv_path, gameweek=gameweek, budget=budget, exclude_injured=True,
                            try_multiple_solvers_flag=False)

# Option 3: Compare solver performance across different strategies
def solver_strategy_comparison(gameweek=0):
    """Compare how different solvers perform across different strategies"""
    strategies = [
        (95.0, 0.3, 0.2, 1.5, "Budget"),
        (100.0, 0.5, 0.3, 2.0, "Standard"), 
        (100.0, 0.7, 0.5, 3.0, "Strong Bench")
    ]
    
    for budget, gk_weight, outfield_weight, min_points, label in strategies:
        print(f"\n{'='*70}")
        print(f"{label.upper()} STRATEGY - £{budget}M BUDGET")
        print(f"{'='*70}")
        squad = optimize_fpl_squad(f'./data_2025/player_expected_scores_gw{gameweek}.csv', 
                                 gameweek=gameweek, budget=budget,
                                 bench_gk_weight=gk_weight, bench_outfield_weight=outfield_weight,
                                 min_bench_outfield_points=min_points,
                                 try_multiple_solvers_flag=True, solver_verbose=True)

# Run optimization
print("🏆 FPL Squad Optimizer - Using Expected Points for Next 4 Gameweeks")
print("=" * 70)

squad = optimize_with_expected_points(gameweek=0)

#%%
# Optional: Run budget comparison
# budget_comparison(gameweek=0)

#%%
# Optional: Show top alternatives by position
def show_alternatives(squad, gameweek=0, top_n=3):
    """Show top alternatives for each position not in squad"""
    all_players = pd.read_csv(f'./data_2025/player_expected_scores_gw{gameweek}.csv')
    
    if 'injury_status' in all_players.columns:
        all_players = all_players[all_players['injury_status'] == 'available']
    
    selected_ids = squad['element_id'].tolist()
    alternatives = all_players[~all_players['element_id'].isin(selected_ids)]
    
    print(f"\n🔄 TOP ALTERNATIVES BY POSITION:")
    print("=" * 50)
    
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_alternatives = alternatives[alternatives['element_type'] == pos].head(top_n)
        print(f"\n{pos}:")
        for _, player in pos_alternatives.iterrows():
            print(f"  {player['web_name']:20} (ID: {player['element_id']}) £{player['cost_per_million']:.1f}m - {player['expected_points_next_4gw']:.1f} pts - Value: {player['value_score']:.1f}")

# Show alternatives
show_alternatives(squad, gameweek=0)
#%%