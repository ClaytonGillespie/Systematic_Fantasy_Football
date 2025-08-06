#!/usr/bin/env python3
"""
FPL Single Period Optimizer 2025

Optimizes FPL teams for a single period using 4-gameweek predictions (GW 1-4).
This approach sums the predictions across gameweeks 1-4 to select the optimal
team for the opening fixtures, allowing for recalibration after GW1 data.
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
    
    print(f"📊 Loading predictions from: {latest_file}")
    
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
    
    print(f"✅ Loaded {len(players_df)} players with 4-gameweek predictions")
    print(f"📈 Position breakdown: {players_df['position_name'].value_counts().to_dict()}")
    
    return players_df

def optimize_single_period_fpl(players_df, budget=100.0, captain_selection=True):
    """
    Single period FPL optimization using 4-gameweek predictions
    
    Args:
        players_df: DataFrame with player data including predicted_4gw
        budget: Budget in millions (default 100.0)
        captain_selection: Whether to include captain optimization
    """
    
    print(f"🎯 Single period optimization for GW 1-4 (budget: £{budget}m)")
    
    # Create the optimization problem
    prob = pulp.LpProblem("FPL_Single_Period", pulp.LpMaximize)
    
    # Decision variables
    squad_vars = {}  # 15-player squad
    starting_vars = {}  # 11-player starting XI
    captain_vars = {} if captain_selection else None  # Captain selection
    
    for idx, player in players_df.iterrows():
        squad_vars[idx] = pulp.LpVariable(f"squad_{idx}", cat='Binary')
        starting_vars[idx] = pulp.LpVariable(f"starting_{idx}", cat='Binary')
        if captain_selection:
            captain_vars[idx] = pulp.LpVariable(f"captain_{idx}", cat='Binary')
    
    # Objective function: Maximize expected points over 4 gameweeks
    objective = []
    
    for idx, player in players_df.iterrows():
        # Starting players get full points
        objective.append(player['predicted_4gw'] * starting_vars[idx])
        
        # Captain gets double points (additional points on top of starting)
        if captain_selection:
            objective.append(player['predicted_4gw'] * captain_vars[idx])
    
    prob += pulp.lpSum(objective)
    
    # Constraints
    
    # 1. Squad size: exactly 15 players
    prob += pulp.lpSum([squad_vars[idx] for idx in players_df.index]) == 15
    
    # 2. Starting XI: exactly 11 players
    prob += pulp.lpSum([starting_vars[idx] for idx in players_df.index]) == 11
    
    # 3. Captain: exactly 1 captain (if enabled)
    if captain_selection:
        prob += pulp.lpSum([captain_vars[idx] for idx in players_df.index]) == 1
    
    # 4. Budget constraint
    prob += pulp.lpSum([squad_vars[idx] * players_df.loc[idx, 'cost_million'] 
                       for idx in players_df.index]) <= budget
    
    # 5. Position constraints for squad (15 players total)
    position_requirements = {1: 2, 2: 5, 3: 5, 4: 3}  # GKP:2, DEF:5, MID:5, FWD:3
    for position, count in position_requirements.items():
        position_players = players_df[players_df['element_type'] == position].index
        prob += pulp.lpSum([squad_vars[idx] for idx in position_players]) == count
    
    # 6. Position constraints for starting XI (flexible formation)
    starting_requirements = {
        1: (1, 1),   # Exactly 1 GKP
        2: (3, 5),   # 3-5 DEF
        3: (2, 5),   # 2-5 MID  
        4: (1, 3)    # 1-3 FWD
    }
    
    for position, (min_count, max_count) in starting_requirements.items():
        position_players = players_df[players_df['element_type'] == position].index
        prob += pulp.lpSum([starting_vars[idx] for idx in position_players]) >= min_count
        prob += pulp.lpSum([starting_vars[idx] for idx in position_players]) <= max_count
    
    # 7. Can only start players who are in the squad
    for idx in players_df.index:
        prob += starting_vars[idx] <= squad_vars[idx]
    
    # 8. Can only captain starting players
    if captain_selection:
        for idx in players_df.index:
            prob += captain_vars[idx] <= starting_vars[idx]
    
    # 9. Team constraints (maximum 3 players from same team)
    teams = players_df['team_short'].unique()
    for team in teams:
        team_players = players_df[players_df['team_short'] == team].index
        prob += pulp.lpSum([squad_vars[idx] for idx in team_players]) <= 3
    
    print("🔄 Solving single period optimization...")
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Check if solution was found
    if prob.status != pulp.LpStatusOptimal:
        print(f"❌ Optimization failed with status: {pulp.LpStatus[prob.status]}")
        return None
    
    print("✅ Optimal solution found!")
    
    # Extract the solution
    squad = []
    starting_xi = []
    captain = None
    total_cost = 0
    total_points = 0
    
    for idx, player in players_df.iterrows():
        if squad_vars[idx].varValue == 1:
            player_info = player.copy()
            player_info['is_starting'] = starting_vars[idx].varValue == 1
            player_info['is_captain'] = (captain_vars[idx].varValue == 1) if captain_selection else False
            
            squad.append(player_info)
            total_cost += player['cost_million']
            
            if player_info['is_starting']:
                starting_xi.append(player_info)
                points = player['predicted_4gw']
                if player_info['is_captain']:
                    points *= 2  # Double points for captain
                    captain = player_info
                total_points += points
    
    print(f"💰 Total squad cost: £{total_cost:.1f}m")
    print(f"🎯 Expected points (GW 1-4): {total_points:.1f}")
    print(f"📊 Expected points per gameweek: {total_points/4:.1f}")
    
    if captain is not None:
        print(f"👑 Captain: {captain['player_name']} ({captain['team_short']}) - {captain['predicted_4gw']:.1f} pts")
    
    return {
        'squad': squad,
        'starting_xi': starting_xi,
        'captain': captain,
        'total_cost': total_cost,
        'total_points': total_points
    }

def display_solution(solution):
    """Display the single period solution"""
    
    if not solution:
        return
    
    print("\n" + "="*80)
    print("🏆 SINGLE PERIOD FPL SOLUTION (GW 1-4)")
    print("="*80)
    
    squad = solution['squad']
    starting_xi = solution['starting_xi']
    captain = solution['captain']
    
    # Show starting XI
    print(f"\n⚽ STARTING XI:")
    print("-" * 100)
    print(f"{'Player':<25} {'Team':<5} {'Pos':<4} {'Cost':<7} {'4GW Pts':<8} {'PPG':<6} {'Value':<6} {'Role'}")
    print("-" * 100)
    
    position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Sort starting XI by position and points
    starting_sorted = sorted(starting_xi, key=lambda x: (x['element_type'], -x['predicted_4gw']))
    
    for player in starting_sorted:
        pos_short = position_names.get(player['element_type'], 'UNK')
        value_per_mil = player['predicted_4gw'] / player['cost_million']
        role = "Captain" if player['is_captain'] else "Starter"
        
        print(f"{player['player_name'][:24]:<25} {player['team_short']:<5} {pos_short:<4} "
              f"£{player['cost_million']:<6.1f} {player['predicted_4gw']:<8.1f} "
              f"{player['predicted_ppg']:<6.2f} {value_per_mil:<6.2f} {role}")
    
    # Show bench
    bench = [p for p in squad if not p['is_starting']]
    if bench:
        print(f"\n🪑 BENCH:")
        print("-" * 100)
        print(f"{'Player':<25} {'Team':<5} {'Pos':<4} {'Cost':<7} {'4GW Pts':<8} {'PPG':<6} {'Value':<6} {'Role'}")
        print("-" * 100)
        
        bench_sorted = sorted(bench, key=lambda x: (x['element_type'], -x['predicted_4gw']))
        
        for player in bench_sorted:
            pos_short = position_names.get(player['element_type'], 'UNK')
            value_per_mil = player['predicted_4gw'] / player['cost_million']
            
            print(f"{player['player_name'][:24]:<25} {player['team_short']:<5} {pos_short:<4} "
                  f"£{player['cost_million']:<6.1f} {player['predicted_4gw']:<8.1f} "
                  f"{player['predicted_ppg']:<6.2f} {value_per_mil:<6.2f} Bench")
    
    # Team analysis
    print(f"\n📊 TEAM ANALYSIS:")
    print("-" * 60)
    
    squad_df = pd.DataFrame(squad)
    
    # Position breakdown
    print("Squad composition:")
    for pos_id, pos_name in position_names.items():
        pos_players = squad_df[squad_df['element_type'] == pos_id]
        if not pos_players.empty:
            count = len(pos_players)
            total_cost = pos_players['cost_million'].sum()
            total_pts = pos_players['predicted_4gw'].sum()
            avg_pts = pos_players['predicted_4gw'].mean()
            
            print(f"  {pos_name}: {count} players, £{total_cost:.1f}m, "
                  f"{total_pts:.1f} total pts, {avg_pts:.1f} avg pts")
    
    # Team distribution
    print(f"\nTeam distribution:")
    team_dist = squad_df['team_short'].value_counts()
    for team, count in team_dist.items():
        players_list = squad_df[squad_df['team_short'] == team]['player_name'].tolist()
        player_names = [name.split()[-1] for name in players_list]  # Just surnames
        print(f"  {team}: {count} players - {', '.join(player_names)}")
    
    # Summary
    print(f"\n💰 Total cost: £{solution['total_cost']:.1f}m")
    print(f"🎯 Expected points over 4 gameweeks: {solution['total_points']:.1f}")
    print(f"📈 Expected points per gameweek: {solution['total_points']/4:.1f}")
    
    if captain is not None:
        captain_contribution = captain['predicted_4gw']  # Additional points from captaincy
        print(f"👑 Captain bonus: +{captain_contribution:.1f} points")
    
    return squad_df

def save_solution(solution, filename_prefix="single_period_team"):
    """Save the single period solution to CSV"""
    
    if not solution:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"data_2025/predictions/{filename_prefix}_{timestamp}.csv"
    
    # Create detailed output
    squad_data = []
    for player in solution['squad']:
        squad_data.append({
            'player_name': player['player_name'],
            'team_short': player['team_short'],
            'position_name': player['position_name'],
            'element_type': player['element_type'],
            'cost_million': player['cost_million'],
            'predicted_4gw': player['predicted_4gw'],
            'predicted_ppg': player.get('predicted_ppg', 0),
            'is_starting': player['is_starting'],
            'is_captain': player['is_captain'],
            'role': 'Captain' if player['is_captain'] else ('Starter' if player['is_starting'] else 'Bench'),
            'optimization_type': 'single_period_gw1_4'
        })
    
    squad_df = pd.DataFrame(squad_data)
    squad_df.to_csv(filename, index=False)
    
    print(f"\n💾 Single period team saved to: {filename}")
    return filename

def main():
    """Main single period optimization pipeline"""
    
    print("🚀 FPL SINGLE PERIOD OPTIMIZER 2025")
    print("⚽ Optimizing for gameweeks 1-4 combined")
    print("="*80)
    
    try:
        # Load predictions
        players_df = load_latest_predictions()
        
        # Run single period optimization
        solution = optimize_single_period_fpl(
            players_df,
            budget=100.0,
            captain_selection=True
        )
        
        if solution:
            # Display results
            squad_df = display_solution(solution)
            
            # Save solution
            save_solution(solution)
            
            # Show top alternatives not selected
            selected_ids = [p['id'] for p in solution['squad']]
            alternatives = players_df[~players_df['id'].isin(selected_ids)]
            
            print(f"\n💡 TOP ALTERNATIVES NOT SELECTED:")
            print("-" * 80)
            print(f"{'Player':<25} {'Team':<5} {'Pos':<4} {'Cost':<7} {'4GW Pts':<8} {'Value'}")
            print("-" * 80)
            
            top_alternatives = alternatives.nlargest(10, 'predicted_4gw')
            position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            
            for _, player in top_alternatives.iterrows():
                pos_short = position_names.get(player['element_type'], 'UNK')
                value_per_mil = player['predicted_4gw'] / player['cost_million']
                print(f"{player['player_name'][:24]:<25} {player['team_short']:<5} {pos_short:<4} "
                      f"£{player['cost_million']:<6.1f} {player['predicted_4gw']:<8.1f} {value_per_mil:<6.2f}")
            
            print(f"\n🎉 Single period optimization complete!")
            print(f"💡 After GW1, you can recalibrate with updated data")
            
            return solution
        else:
            print("❌ No optimal team found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    solution = main()