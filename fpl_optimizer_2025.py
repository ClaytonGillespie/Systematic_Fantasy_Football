#!/usr/bin/env python3
"""
FPL Optimizer 2025 - Multi-period optimization using 4-gameweek predictions

This optimizer uses the predicted_4gw values from the predictions/ folder to optimize
FPL squads considering transfer costs for multi-period planning.
"""

import pandas as pd
import pulp
import numpy as np
from datetime import datetime
import glob
import os

def load_latest_predictions():
    """Load the latest prediction file from data_2025/predictions/"""
    
    # Find the latest prediction file
    prediction_files = glob.glob("data_2025/predictions/fantasy_predictions_2025_final_*.csv")
    
    if not prediction_files:
        raise FileNotFoundError("No prediction files found in data_2025/predictions/")
    
    # Get the most recent file
    latest_file = max(prediction_files, key=os.path.getctime)
    
    print(f"Loading predictions from: {latest_file}")
    
    # Load the data
    players_df = pd.read_csv(latest_file)
    
    # Clean and prepare the data
    players_df = players_df.dropna(subset=['predicted_4gw', 'cost_million', 'position_name'])
    
    # Map position names to numbers for constraints
    position_mapping = {
        'Goalkeeper': 1,
        'Defender': 2, 
        'Midfielder': 3,
        'Forward': 4
    }
    
    players_df['element_type'] = players_df['position_name'].map(position_mapping)
    players_df = players_df.dropna(subset=['element_type'])
    
    print(f"✅ Loaded {len(players_df)} players with predictions")
    print(f"Position breakdown: {players_df['position_name'].value_counts().to_dict()}")
    
    return players_df

def optimize_fpl_team(players_df, budget=100.0, include_bench_value=True, 
                     bench_gk_weight=0.3, bench_outfield_weight=0.5):
    """
    Optimize FPL team using 4-gameweek predictions
    
    Args:
        players_df: DataFrame with player data including predicted_4gw
        budget: Budget in millions (default 100.0)
        include_bench_value: Whether to include bench player value in optimization
        bench_gk_weight: Weight for bench goalkeeper (0-1)
        bench_outfield_weight: Weight for bench outfield players (0-1)
    """
    
    print(f"🎯 Optimizing FPL team with £{budget}m budget")
    
    # Create the optimization problem
    prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
    
    # Decision variables: 1 if player is selected, 0 otherwise
    player_vars = {}
    for idx, player in players_df.iterrows():
        player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    # Starting XI variables (who actually plays)
    starting_vars = {}
    for idx in players_df.index:
        starting_vars[idx] = pulp.LpVariable(f"starting_{idx}", cat='Binary')
    
    # Objective function: Maximize expected points over 4 gameweeks
    objective = []
    
    for idx, player in players_df.iterrows():
        # Starting players get full points
        objective.append(player['predicted_4gw'] * starting_vars[idx])
        
        # Bench players get reduced value if include_bench_value is True
        if include_bench_value:
            if player['element_type'] == 1:  # Goalkeeper
                bench_value = player['predicted_4gw'] * bench_gk_weight
            else:  # Outfield players
                bench_value = player['predicted_4gw'] * bench_outfield_weight
            
            # Bench value = selected but not starting
            objective.append(bench_value * (player_vars[idx] - starting_vars[idx]))
    
    prob += pulp.lpSum(objective)
    
    # Constraints
    
    # 1. Squad size: exactly 15 players
    prob += pulp.lpSum([player_vars[idx] for idx in players_df.index]) == 15
    
    # 2. Starting XI: exactly 11 players
    prob += pulp.lpSum([starting_vars[idx] for idx in players_df.index]) == 11
    
    # 3. Budget constraint
    prob += pulp.lpSum([player_vars[idx] * players_df.loc[idx, 'cost_million'] 
                       for idx in players_df.index]) <= budget
    
    # 4. Position constraints for squad (15 players)
    for position, (min_count, max_count) in [(1, (2, 2)), (2, (5, 5)), (3, (5, 5)), (4, (3, 3))]:
        position_players = players_df[players_df['element_type'] == position].index
        prob += pulp.lpSum([player_vars[idx] for idx in position_players]) == max_count
    
    # 5. Position constraints for starting XI
    for position, (min_count, max_count) in [(1, (1, 1)), (2, (3, 5)), (3, (2, 5)), (4, (1, 3))]:
        position_players = players_df[players_df['element_type'] == position].index
        prob += pulp.lpSum([starting_vars[idx] for idx in position_players]) >= min_count
        prob += pulp.lpSum([starting_vars[idx] for idx in position_players]) <= max_count
    
    # 6. Can only start players who are in the squad
    for idx in players_df.index:
        prob += starting_vars[idx] <= player_vars[idx]
    
    # 7. Team constraints (maximum 3 players from same team)
    teams = players_df['team_short'].unique()
    for team in teams:
        team_players = players_df[players_df['team_short'] == team].index
        prob += pulp.lpSum([player_vars[idx] for idx in team_players]) <= 3
    
    print("🔄 Solving optimization problem...")
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Check if solution was found
    if prob.status != pulp.LpStatusOptimal:
        print(f"❌ Optimization failed with status: {pulp.LpStatus[prob.status]}")
        return None, None
    
    print("✅ Optimal solution found!")
    
    # Extract the solution
    selected_players = []
    starting_players = []
    
    total_cost = 0
    total_predicted_points = 0
    
    for idx, player in players_df.iterrows():
        if player_vars[idx].varValue == 1:
            player_info = player.copy()
            player_info['is_starting'] = starting_vars[idx].varValue == 1
            player_info['role'] = 'Starting XI' if player_info['is_starting'] else 'Bench'
            selected_players.append(player_info)
            total_cost += player['cost_million']
            
            if player_info['is_starting']:
                starting_players.append(player_info)
                total_predicted_points += player['predicted_4gw']
            else:
                # Add reduced bench value
                if player['element_type'] == 1:  # GK
                    total_predicted_points += player['predicted_4gw'] * bench_gk_weight
                else:
                    total_predicted_points += player['predicted_4gw'] * bench_outfield_weight
    
    print(f"\n💰 Total cost: £{total_cost:.1f}m (Budget: £{budget}m)")
    print(f"🎯 Expected points (4GW): {total_predicted_points:.1f}")
    print(f"📊 Expected points per gameweek: {total_predicted_points/4:.1f}")
    
    return selected_players, starting_players

def analyze_team(selected_players):
    """Analyze the selected team"""
    
    if not selected_players:
        return
    
    selected_df = pd.DataFrame(selected_players)
    
    print("\n" + "="*80)
    print("🏆 OPTIMAL FPL TEAM (4 GAMEWEEKS)")
    print("="*80)
    
    # Starting XI
    starting_xi = selected_df[selected_df['is_starting'] == True].sort_values(
        ['element_type', 'predicted_4gw'], ascending=[True, False]
    )
    
    print(f"\n⚽ STARTING XI:")
    print("-" * 90)
    print(f"{'Player':<25} {'Team':<5} {'Pos':<4} {'Cost':<6} {'4GW Pts':<8} {'PPG':<6} {'Last PPG':<8} {'Value':<6}")
    print("-" * 90)
    
    position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    for _, player in starting_xi.iterrows():
        pos_short = position_names.get(player['element_type'], 'UNK')
        value_per_mil = player['predicted_4gw'] / player['cost_million']
        
        print(f"{player['player_name'][:24]:<25} {player['team_short']:<5} {pos_short:<4} "
              f"£{player['cost_million']:<5.1f} {player['predicted_4gw']:<8.1f} "
              f"{player['predicted_ppg']:<6.2f} {player['last_season_ppg']:<8.2f} {value_per_mil:<6.2f}")
    
    # Bench
    bench = selected_df[selected_df['is_starting'] == False].sort_values(
        ['element_type', 'predicted_4gw'], ascending=[True, False]
    )
    
    print(f"\n🪑 BENCH:")
    print("-" * 90)
    print(f"{'Player':<25} {'Team':<5} {'Pos':<4} {'Cost':<6} {'4GW Pts':<8} {'PPG':<6} {'Last PPG':<8} {'Value':<6}")
    print("-" * 90)
    
    for _, player in bench.iterrows():
        pos_short = position_names.get(player['element_type'], 'UNK')
        value_per_mil = player['predicted_4gw'] / player['cost_million']
        
        print(f"{player['player_name'][:24]:<25} {player['team_short']:<5} {pos_short:<4} "
              f"£{player['cost_million']:<5.1f} {player['predicted_4gw']:<8.1f} "
              f"{player['predicted_ppg']:<6.2f} {player['last_season_ppg']:<8.2f} {value_per_mil:<6.2f}")
    
    # Team analysis
    print(f"\n📊 TEAM ANALYSIS:")
    print("-" * 50)
    
    # Position breakdown
    position_analysis = selected_df.groupby('element_type').agg({
        'predicted_4gw': ['count', 'sum', 'mean'],
        'cost_million': 'sum'
    }).round(2)
    
    print("Position breakdown:")
    for pos_id, pos_name in position_names.items():
        if pos_id in position_analysis.index:
            stats = position_analysis.loc[pos_id]
            count = int(stats[('predicted_4gw', 'count')])
            total_pts = stats[('predicted_4gw', 'sum')]
            avg_pts = stats[('predicted_4gw', 'mean')]
            total_cost = stats[('cost_million', 'sum')]
            
            print(f"  {pos_name}: {count} players, £{total_cost:.1f}m, "
                  f"{total_pts:.1f} total pts, {avg_pts:.1f} avg pts")
    
    # Team distribution
    print(f"\nTeam distribution:")
    team_dist = selected_df['team_short'].value_counts()
    for team, count in team_dist.items():
        players_list = selected_df[selected_df['team_short'] == team]['player_name'].tolist()
        print(f"  {team}: {count} players - {', '.join([p.split()[-1] for p in players_list])}")
    
    return selected_df

def save_team(selected_players, filename_prefix="optimal_fpl_team"):
    """Save the optimal team to CSV"""
    
    if not selected_players:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"data_2025/predictions/{filename_prefix}_{timestamp}.csv"
    
    selected_df = pd.DataFrame(selected_players)
    selected_df.to_csv(filename, index=False)
    
    print(f"\n💾 Team saved to: {filename}")
    return filename

def main():
    """Main optimization pipeline"""
    
    print("🚀 FPL OPTIMIZER 2025 - 4 GAMEWEEK PREDICTIONS")
    print("="*80)
    
    try:
        # Load predictions
        players_df = load_latest_predictions()
        
        # Run optimization with different budget scenarios
        budgets = [100.0, 98.0, 96.0]  # Different budget scenarios
        
        best_team = None
        best_score = 0
        
        for budget in budgets:
            print(f"\n{'='*20} BUDGET: £{budget}M {'='*20}")
            
            selected_players, starting_players = optimize_fpl_team(
                players_df, 
                budget=budget,
                include_bench_value=True,
                bench_gk_weight=0.3,
                bench_outfield_weight=0.5
            )
            
            if selected_players:
                # Calculate total expected score
                total_score = sum(p['predicted_4gw'] for p in starting_players)
                bench_score = sum(p['predicted_4gw'] * (0.3 if p['element_type'] == 1 else 0.5) 
                                for p in selected_players if not p['is_starting'])
                total_score += bench_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_team = selected_players
                
                print(f"📈 Total expected score: {total_score:.1f} points")
        
        # Analyze and save the best team
        if best_team:
            print(f"\n🏆 BEST TEAM FOUND:")
            team_df = analyze_team(best_team)
            save_team(best_team)
            
            # Show top alternatives not selected
            selected_ids = [p['id'] for p in best_team]
            alternatives = players_df[~players_df['id'].isin(selected_ids)]
            
            print(f"\n💡 TOP ALTERNATIVES NOT SELECTED:")
            print("-" * 70)
            top_alternatives = alternatives.nlargest(10, 'predicted_4gw')
            
            for _, player in top_alternatives.iterrows():
                pos_short = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['element_type'], 'UNK')
                print(f"{player['player_name'][:20]:<20} ({player['team_short']}) {pos_short} "
                      f"£{player['cost_million']:.1f}m - {player['predicted_4gw']:.1f} pts")
            
            return team_df
        else:
            print("❌ No optimal team found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    optimal_team = main()