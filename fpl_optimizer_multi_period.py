#!/usr/bin/env python3
"""
FPL Multi-Period Optimizer 2025

True multi-period optimization considering:
1. Individual gameweek fixtures (GW1-4)
2. Captain selection with double points
3. Free captain changes between gameweeks
4. Transfer costs for squad changes
"""

import pandas as pd
import pulp
import numpy as np
from datetime import datetime
import glob
import os

def load_fixture_data():
    """Load fixture data to understand opponent difficulty for each gameweek"""
    
    try:
        fixtures_df = pd.read_csv("data_2025/fixtures_gw0.csv")
        teams_df = pd.read_csv("data_2025/teams_gw0.csv")
        
        print(f"✅ Loaded {len(fixtures_df)} fixtures")
        
        # For this example, we'll create a simple fixture difficulty matrix
        # In a real implementation, you'd parse the actual fixture list
        
        # Create team difficulty mapping
        team_strength = dict(zip(teams_df['short_name'], teams_df['strength']))
        
        return fixtures_df, teams_df, team_strength
        
    except FileNotFoundError:
        print("⚠️ Fixture data not found, using default difficulties")
        return None, None, {}

def estimate_gameweek_points(player_4gw_points, base_difficulty=3):
    """
    Estimate individual gameweek points from 4GW aggregate
    
    Args:
        player_4gw_points: Total points over 4 gameweeks
        base_difficulty: Base fixture difficulty (1=easy, 5=hard)
    
    Returns:
        List of 4 gameweek point estimates
    """
    
    # Simple approach: distribute 4GW points across gameweeks with some variation
    base_points_per_gw = player_4gw_points / 4.0
    
    # Add some realistic variation (±20% typically)
    np.random.seed(42)  # For reproducible results
    variations = np.random.normal(1.0, 0.15, 4)
    variations = np.clip(variations, 0.5, 1.8)  # Reasonable bounds
    
    gw_points = [base_points_per_gw * var for var in variations]
    
    # Normalize to ensure sum equals original 4GW total
    current_sum = sum(gw_points)
    gw_points = [pts * player_4gw_points / current_sum for pts in gw_points]
    
    return gw_points

def optimize_multi_period_fpl(players_df, num_gameweeks=4, budget=100.0, 
                             transfer_cost_points=4, free_transfers_per_gw=1):
    """
    Multi-period FPL optimization with captain selection
    
    Args:
        players_df: Player data with predictions
        num_gameweeks: Number of gameweeks to optimize
        budget: Starting budget
        transfer_cost: Cost per transfer beyond free transfers
        free_transfers_per_gw: Free transfers per gameweek
    """
    
    print(f"🎯 Multi-period optimization: {num_gameweeks} gameweeks")
    print(f"💰 Budget: £{budget}m, Transfer cost: {transfer_cost_points} points")
    
    # Prepare gameweek-specific data
    players_gw_data = {}
    
    for gw in range(1, num_gameweeks + 1):
        players_gw_data[gw] = []
        
        for idx, player in players_df.iterrows():
            # Estimate points for this specific gameweek
            gw_points_list = estimate_gameweek_points(player['predicted_4gw'])
            gw_points = gw_points_list[gw - 1]  # 0-indexed
            
            players_gw_data[gw].append({
                'id': idx,
                'player_name': player['player_name'],
                'team': player['team_short'],
                'position': player['element_type'],
                'cost': player['cost_million'],
                'points': max(0, gw_points)  # Ensure non-negative
            })
    
    # Create optimization problem
    prob = pulp.LpProblem("FPL_Multi_Period", pulp.LpMaximize)
    
    # Decision Variables
    
    # Squad selection: x[i,t] = 1 if player i is in squad in gameweek t
    squad_vars = {}
    for gw in range(1, num_gameweeks + 1):
        squad_vars[gw] = {}
        for player_data in players_gw_data[gw]:
            player_id = player_data['id']
            squad_vars[gw][player_id] = pulp.LpVariable(f"squad_{player_id}_gw{gw}", cat='Binary')
    
    # Starting XI: s[i,t] = 1 if player i starts in gameweek t
    starting_vars = {}
    for gw in range(1, num_gameweeks + 1):
        starting_vars[gw] = {}
        for player_data in players_gw_data[gw]:
            player_id = player_data['id']
            starting_vars[gw][player_id] = pulp.LpVariable(f"start_{player_id}_gw{gw}", cat='Binary')
    
    # Captain selection: c[i,t] = 1 if player i is captain in gameweek t
    captain_vars = {}
    for gw in range(1, num_gameweeks + 1):
        captain_vars[gw] = {}
        for player_data in players_gw_data[gw]:
            player_id = player_data['id']
            captain_vars[gw][player_id] = pulp.LpVariable(f"captain_{player_id}_gw{gw}", cat='Binary')
    
    # Transfer variables: transfers_in[i,t] = 1 if player i is transferred in for gameweek t
    transfers_in = {}
    transfers_out = {}
    
    for gw in range(2, num_gameweeks + 1):  # Transfers start from GW2
        transfers_in[gw] = {}
        transfers_out[gw] = {}
        for player_data in players_gw_data[gw]:
            player_id = player_data['id']
            transfers_in[gw][player_id] = pulp.LpVariable(f"transfer_in_{player_id}_gw{gw}", cat='Binary')
            transfers_out[gw][player_id] = pulp.LpVariable(f"transfer_out_{player_id}_gw{gw}", cat='Binary')
    
    # Objective Function: Maximize total points across all gameweeks
    objective = []
    
    for gw in range(1, num_gameweeks + 1):
        for player_data in players_gw_data[gw]:
            player_id = player_data['id']
            points = player_data['points']
            
            # Regular points for starting players
            objective.append(points * starting_vars[gw][player_id])
            
            # Double points for captain (additional points on top of regular points)
            objective.append(points * captain_vars[gw][player_id])
    
    # Subtract transfer costs (in points, not money)
    for gw in range(2, num_gameweeks + 1):
        total_transfers = pulp.lpSum([transfers_in[gw][pid] for pid in transfers_in[gw].keys()])
        # Points deducted for transfers beyond free transfers
        transfer_penalty_points = pulp.LpVariable(f"transfer_penalty_gw{gw}", lowBound=0)
        objective.append(-transfer_penalty_points)
        
        # Link transfer penalty to actual transfers (4 points per extra transfer)
        prob += transfer_penalty_points >= (total_transfers - free_transfers_per_gw) * transfer_cost_points
    
    prob += pulp.lpSum(objective)
    
    # Constraints
    
    # 1. Squad size constraints (15 players each gameweek)
    for gw in range(1, num_gameweeks + 1):
        prob += pulp.lpSum([squad_vars[gw][pid] for pid in squad_vars[gw].keys()]) == 15
    
    # 2. Starting XI constraints (11 players each gameweek)
    for gw in range(1, num_gameweeks + 1):
        prob += pulp.lpSum([starting_vars[gw][pid] for pid in starting_vars[gw].keys()]) == 11
    
    # 3. Captain constraints (exactly 1 captain per gameweek)
    for gw in range(1, num_gameweeks + 1):
        prob += pulp.lpSum([captain_vars[gw][pid] for pid in captain_vars[gw].keys()]) == 1
    
    # 4. Position constraints for squad
    for gw in range(1, num_gameweeks + 1):
        position_counts = {1: 2, 2: 5, 3: 5, 4: 3}  # GKP: 2, DEF: 5, MID: 5, FWD: 3
        for pos, count in position_counts.items():
            pos_players = [pid for pid in squad_vars[gw].keys() 
                          if next(p for p in players_gw_data[gw] if p['id'] == pid)['position'] == pos]
            prob += pulp.lpSum([squad_vars[gw][pid] for pid in pos_players]) == count
    
    # 5. Position constraints for starting XI
    for gw in range(1, num_gameweeks + 1):
        # Flexible formation constraints
        pos_constraints = {
            1: (1, 1),   # Exactly 1 GKP
            2: (3, 5),   # 3-5 DEF
            3: (2, 5),   # 2-5 MID  
            4: (1, 3)    # 1-3 FWD
        }
        
        for pos, (min_count, max_count) in pos_constraints.items():
            pos_players = [pid for pid in starting_vars[gw].keys()
                          if next(p for p in players_gw_data[gw] if p['id'] == pid)['position'] == pos]
            prob += pulp.lpSum([starting_vars[gw][pid] for pid in pos_players]) >= min_count
            prob += pulp.lpSum([starting_vars[gw][pid] for pid in pos_players]) <= max_count
    
    # 6. Can only start players who are in squad
    for gw in range(1, num_gameweeks + 1):
        for pid in squad_vars[gw].keys():
            prob += starting_vars[gw][pid] <= squad_vars[gw][pid]
    
    # 7. Can only captain starting players
    for gw in range(1, num_gameweeks + 1):
        for pid in captain_vars[gw].keys():
            prob += captain_vars[gw][pid] <= starting_vars[gw][pid]
    
    # 8. Budget constraint (initial squad only)
    first_gw_cost = pulp.lpSum([
        squad_vars[1][pid] * next(p for p in players_gw_data[1] if p['id'] == pid)['cost']
        for pid in squad_vars[1].keys()
    ])
    prob += first_gw_cost <= budget
    
    # 9. Team constraints (max 3 players from same team)
    for gw in range(1, num_gameweeks + 1):
        teams = set(p['team'] for p in players_gw_data[gw])
        for team in teams:
            team_players = [pid for pid in squad_vars[gw].keys()
                           if next(p for p in players_gw_data[gw] if p['id'] == pid)['team'] == team]
            prob += pulp.lpSum([squad_vars[gw][pid] for pid in team_players]) <= 3
    
    # 10. Squad continuity constraints (transfers)
    for gw in range(2, num_gameweeks + 1):
        prev_gw = gw - 1
        for pid in squad_vars[gw].keys():
            if pid in squad_vars[prev_gw]:
                # Squad continuity: player is in squad[gw] if (in squad[prev_gw] and not transferred out) or transferred in
                prob += (squad_vars[gw][pid] == 
                        squad_vars[prev_gw][pid] - transfers_out[gw].get(pid, 0) + transfers_in[gw].get(pid, 0))
    
    print("🔄 Solving multi-period optimization...")
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        print(f"❌ Optimization failed: {pulp.LpStatus[prob.status]}")
        return None
    
    print("✅ Multi-period solution found!")
    
    # Extract solution
    solution = {}
    total_points = 0
    
    for gw in range(1, num_gameweeks + 1):
        gw_squad = []
        gw_starting = []
        gw_captain = None
        
        for player_data in players_gw_data[gw]:
            pid = player_data['id']
            
            if squad_vars[gw][pid].varValue == 1:
                player_info = player_data.copy()
                player_info['is_starting'] = starting_vars[gw][pid].varValue == 1
                player_info['is_captain'] = captain_vars[gw][pid].varValue == 1
                
                if player_info['is_captain']:
                    gw_captain = player_info
                    total_points += player_info['points'] * 2  # Double points
                elif player_info['is_starting']:
                    total_points += player_info['points']
                    
                gw_squad.append(player_info)
                
                if player_info['is_starting']:
                    gw_starting.append(player_info)
        
        solution[gw] = {
            'squad': gw_squad,
            'starting_xi': gw_starting,
            'captain': gw_captain
        }
    
    print(f"🎯 Total expected points: {total_points:.1f}")
    print(f"📊 Average per gameweek: {total_points/num_gameweeks:.1f}")
    
    return solution

def display_multi_period_solution(solution, num_gameweeks=4):
    """Display the multi-period solution with initial team + transfers"""
    
    if not solution:
        return
    
    print("\n" + "="*80)
    print("🏆 MULTI-PERIOD FPL SOLUTION")
    print("="*80)
    
    # Show initial team (GW1)
    initial_squad = solution[1]['squad']
    initial_starting = solution[1]['starting_xi']
    initial_captain = solution[1]['captain']
    
    print(f"\n🏁 INITIAL TEAM (GAMEWEEK 1)")
    print("-" * 60)
    
    # Show starting XI
    print(f"\n⚽ STARTING XI:")
    starting_xi = sorted(initial_starting, key=lambda x: (x['position'], -x['points']))
    
    for player in starting_xi:
        pos_name = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['position'], 'UNK')
        captain_marker = " (C)" if player['is_captain'] else ""
        print(f"  {player['player_name']:<25} ({player['team']}) {pos_name} £{player['cost']:.1f}m{captain_marker}")
    
    # Show bench
    bench = [p for p in initial_squad if not p['is_starting']]
    if bench:
        print(f"\n🪑 BENCH:")
        for player in sorted(bench, key=lambda x: (x['position'], -x['points'])):
            pos_name = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['position'], 'UNK')
            print(f"  {player['player_name']:<25} ({player['team']}) {pos_name} £{player['cost']:.1f}m")
    
    # Calculate and show initial cost
    total_cost = sum(p['cost'] for p in initial_squad)
    print(f"\n💰 Initial squad cost: £{total_cost:.1f}m")
    
    if initial_captain:
        print(f"👑 Initial captain: {initial_captain['player_name']} ({initial_captain['team']})")
    
    # Show transfers and captain changes for subsequent gameweeks
    for gw in range(2, num_gameweeks + 1):
        current_squad = {p['player_name'] for p in solution[gw]['squad']}
        previous_squad = {p['player_name'] for p in solution[gw-1]['squad']}
        
        # Find transfers
        transfers_out = previous_squad - current_squad
        transfers_in = current_squad - previous_squad
        
        current_captain = solution[gw]['captain']
        previous_captain = solution[gw-1]['captain']
        
        # Only show GW if there are changes
        if transfers_out or transfers_in or (current_captain['player_name'] != previous_captain['player_name']):
            print(f"\n⚽ GAMEWEEK {gw} CHANGES")
            print("-" * 40)
            
            # Show transfers
            if transfers_out or transfers_in:
                transfers_out_list = list(transfers_out)
                transfers_in_list = list(transfers_in)
                num_transfers = max(len(transfers_out_list), len(transfers_in_list))
                
                if num_transfers > 0:
                    print(f"🔄 TRANSFERS ({num_transfers} transfer{'s' if num_transfers > 1 else ''}):")
                    
                    for i in range(num_transfers):
                        out_player = transfers_out_list[i] if i < len(transfers_out_list) else ""
                        in_player = transfers_in_list[i] if i < len(transfers_in_list) else ""
                        
                        if out_player and in_player:
                            # Find player details
                            out_details = next((p for p in solution[gw-1]['squad'] if p['player_name'] == out_player), None)
                            in_details = next((p for p in solution[gw]['squad'] if p['player_name'] == in_player), None)
                            
                            out_pos = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(out_details['position'], 'UNK') if out_details else 'UNK'
                            in_pos = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(in_details['position'], 'UNK') if in_details else 'UNK'
                            out_cost = f"£{out_details['cost']:.1f}m" if out_details else ""
                            in_cost = f"£{in_details['cost']:.1f}m" if in_details else ""
                            
                            print(f"  OUT: {out_player:<25} ({out_pos}) {out_cost}")
                            print(f"  IN:  {in_player:<25} ({in_pos}) {in_cost}")
                        elif out_player:
                            out_details = next((p for p in solution[gw-1]['squad'] if p['player_name'] == out_player), None)
                            out_pos = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(out_details['position'], 'UNK') if out_details else 'UNK'
                            print(f"  OUT: {out_player:<25} ({out_pos})")
                        elif in_player:
                            in_details = next((p for p in solution[gw]['squad'] if p['player_name'] == in_player), None)
                            in_pos = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(in_details['position'], 'UNK') if in_details else 'UNK'
                            print(f"  IN:  {in_player:<25} ({in_pos})")
                    
                    # Show transfer cost
                    free_transfers = 1  # 1 free transfer per week
                    extra_transfers = max(0, num_transfers - free_transfers)
                    if extra_transfers > 0:
                        points_deducted = extra_transfers * 4
                        print(f"  💸 Transfer cost: -{points_deducted} points ({extra_transfers} extra transfer{'s' if extra_transfers > 1 else ''})")
            
            # Show captain change
            if current_captain and previous_captain and current_captain['player_name'] != previous_captain['player_name']:
                print(f"👑 CAPTAIN CHANGE:")
                print(f"  OUT: {previous_captain['player_name']} ({previous_captain['team']})")
                print(f"  IN:  {current_captain['player_name']} ({current_captain['team']})")
            elif current_captain:
                print(f"👑 CAPTAIN: {current_captain['player_name']} ({current_captain['team']})")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 40)
    
    total_points = 0
    total_captain_points = 0
    total_transfer_penalties = 0
    
    for gw in range(1, num_gameweeks + 1):
        gw_points = sum(p['points'] for p in solution[gw]['starting_xi'])
        captain_bonus = solution[gw]['captain']['points'] if solution[gw]['captain'] else 0
        
        total_points += gw_points + captain_bonus
        total_captain_points += captain_bonus
        
        # Calculate transfer penalties
        if gw > 1:
            current_squad = {p['player_name'] for p in solution[gw]['squad']}
            previous_squad = {p['player_name'] for p in solution[gw-1]['squad']}
            num_transfers = len(current_squad - previous_squad)
            extra_transfers = max(0, num_transfers - 1)  # 1 free transfer
            transfer_penalty = extra_transfers * 4
            total_transfer_penalties += transfer_penalty
            total_points -= transfer_penalty
    
    print(f"Expected total points: {total_points:.1f}")
    print(f"Captain bonus points: +{total_captain_points:.1f}")
    print(f"Transfer penalties: -{total_transfer_penalties:.1f}")
    print(f"Average per gameweek: {total_points/num_gameweeks:.1f}")
    
    return solution

def main():
    """Main multi-period optimization"""
    
    print("🚀 FPL MULTI-PERIOD OPTIMIZER 2025")
    print("="*80)
    
    try:
        # Load latest predictions
        prediction_files = glob.glob("data_2025/predictions/fantasy_predictions_2025_final_*.csv")
        if not prediction_files:
            raise FileNotFoundError("No prediction files found")
        
        latest_file = max(prediction_files, key=os.path.getctime)
        players_df = pd.read_csv(latest_file)
        
        # Clean data
        players_df = players_df.dropna(subset=['predicted_4gw', 'cost_million', 'element_type'])
        players_df = players_df[players_df['element_type'] > 0]  # Valid positions only
        
        print(f"✅ Loaded {len(players_df)} players from {latest_file}")
        
        # Run multi-period optimization
        solution = optimize_multi_period_fpl(
            players_df,
            num_gameweeks=4,
            budget=100.0,
            transfer_cost_points=4,
            free_transfers_per_gw=1
        )
        
        # Display results
        display_multi_period_solution(solution)
        
        # Save solution (initial team + transfers format)
        if solution:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Create transfer-focused summary for saving
            summary_data = []
            
            # 1. Save initial team (GW1)
            initial_squad = solution[1]['squad']
            for player in initial_squad:
                summary_data.append({
                    'gameweek': 1,
                    'action_type': 'initial_squad',
                    'player_name': player['player_name'],
                    'team': player['team'],
                    'position': {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['position'], 'UNK'),
                    'cost': player['cost'],
                    'points': player['points'],
                    'is_starting': player['is_starting'],
                    'is_captain': player['is_captain'],
                    'transfer_details': 'Initial 15-man squad'
                })
            
            # 2. Save transfers and captain changes for GW2-4
            for gw in range(2, 5):  # GW 2,3,4
                current_squad = {p['player_name'] for p in solution[gw]['squad']}
                previous_squad = {p['player_name'] for p in solution[gw-1]['squad']}
                
                # Find transfers
                transfers_out = previous_squad - current_squad
                transfers_in = current_squad - previous_squad
                
                # Record transfers out
                for player_name in transfers_out:
                    player_details = next((p for p in solution[gw-1]['squad'] if p['player_name'] == player_name), None)
                    if player_details:
                        summary_data.append({
                            'gameweek': gw,
                            'action_type': 'transfer_out',
                            'player_name': player_name,
                            'team': player_details['team'],
                            'position': {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player_details['position'], 'UNK'),
                            'cost': player_details['cost'],
                            'points': player_details['points'],
                            'is_starting': False,
                            'is_captain': False,
                            'transfer_details': f'Transferred out in GW{gw}'
                        })
                
                # Record transfers in
                for player_name in transfers_in:
                    player_details = next((p for p in solution[gw]['squad'] if p['player_name'] == player_name), None)
                    if player_details:
                        summary_data.append({
                            'gameweek': gw,
                            'action_type': 'transfer_in',
                            'player_name': player_name,
                            'team': player_details['team'],
                            'position': {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player_details['position'], 'UNK'),
                            'cost': player_details['cost'],
                            'points': player_details['points'],
                            'is_starting': player_details['is_starting'],
                            'is_captain': player_details['is_captain'],
                            'transfer_details': f'Transferred in for GW{gw}'
                        })
                
                # Record captain changes
                current_captain = solution[gw]['captain']
                previous_captain = solution[gw-1]['captain']
                
                if current_captain and previous_captain and current_captain['player_name'] != previous_captain['player_name']:
                    summary_data.append({
                        'gameweek': gw,
                        'action_type': 'captain_change',
                        'player_name': current_captain['player_name'],
                        'team': current_captain['team'],
                        'position': {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(current_captain['position'], 'UNK'),
                        'cost': current_captain['cost'],
                        'points': current_captain['points'],
                        'is_starting': True,
                        'is_captain': True,
                        'transfer_details': f'Made captain in GW{gw} (replacing {previous_captain["player_name"]})'
                    })
            
            summary_df = pd.DataFrame(summary_data)
            filename = f"data_2025/predictions/multi_period_transfers_{timestamp}.csv"
            summary_df.to_csv(filename, index=False)
            
            print(f"\n💾 Transfer-focused solution saved to: {filename}")
        
        return solution
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    solution = main()