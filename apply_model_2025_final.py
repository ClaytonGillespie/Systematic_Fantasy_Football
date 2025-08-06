#!/usr/bin/env python3
"""
Fantasy Football 2025 Predictions Script - Final Version

This version addresses the scaling issues and ensures top performers like Salah 
get appropriate predictions for 4-gameweek periods.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main prediction pipeline"""
    
    print("🚀 FANTASY FOOTBALL 2025 PREDICTION PIPELINE - FINAL")
    print("="*80)
    
    # Load data
    print("Loading data...")
    players_df = pd.read_csv('./data_2025/players_gw0.csv')
    positions_df = pd.read_csv('./data_2025/positions_gw0.csv')
    previous_season_df = pd.read_csv('./data_2025/previous_season_elements.csv')
    
    # The players data already has text position names, so we don't need to map
    # Just ensure we have the position_name column
    if 'position' in players_df.columns:
        # The position column in players_gw0.csv contains text names like "Midfielder"
        players_df['position_name'] = players_df['position']
    else:
        # Fallback: create position mapping if needed
        position_map = dict(zip(positions_df['id'], positions_df['singular_name']))
        players_df['position_name'] = players_df['position'].map(position_map)
    
    print(f"Loaded {len(players_df)} players")
    
    # Match with previous season data
    print("Matching with previous season performance...")
    
    results = []
    
    for _, player in players_df.iterrows():
        # Find previous season match
        prev_matches = previous_season_df[
            (previous_season_df['first_name'].str.lower() == player['first_name'].lower()) &
            (previous_season_df['second_name'].str.lower() == player['second_name'].lower())
        ]
        
        if len(prev_matches) > 0:
            prev_player = prev_matches.iloc[0]
            last_season_total = prev_player['total_points']
        else:
            # For new players, estimate based on cost and position
            cost = player['now_cost'] / 10.0
            position = player['position']
            
            if position == 1:  # GKP
                last_season_total = max(50, cost * 12)
            elif position == 2:  # DEF
                last_season_total = max(40, cost * 10)
            elif position == 3:  # MID
                last_season_total = max(30, cost * 12)
            else:  # FWD
                last_season_total = max(25, cost * 15)
        
        # Calculate 4-gameweek prediction based on last season performance
        # Use a more realistic scaling that puts top players in the right range
        
        base_4gw_prediction = last_season_total / 38 * 4  # 4 games worth of last season average
        
        # Apply some variation based on cost (higher cost = premium players)
        cost_multiplier = 1 + (player['now_cost'] - 40) / 200  # Slight boost for expensive players
        
        # Apply position-specific adjustments
        if player['position'] == 4:  # Forwards
            position_multiplier = 1.1
        elif player['position'] == 3:  # Midfielders
            position_multiplier = 1.05
        else:
            position_multiplier = 1.0
        
        # Final prediction
        predicted_4gw = base_4gw_prediction * cost_multiplier * position_multiplier
        
        # Apply reasonable bounds (negative to 100 over 4 gameweeks)
        predicted_4gw = np.clip(predicted_4gw, -2.0, 100.0)
        
        # Add some controlled randomness to avoid identical predictions
        noise = np.random.normal(0, 0.1)
        predicted_4gw += noise
        predicted_4gw = np.clip(predicted_4gw, -2.0, 100.0)
        
        # Map position names to numbers for optimizer
        position_to_num = {
            'Goalkeeper': 1,
            'Defender': 2,
            'Midfielder': 3,
            'Forward': 4
        }
        
        results.append({
            'id': player['id'],
            'first_name': player['first_name'],
            'second_name': player['second_name'],
            'player_name': f"{player['first_name']} {player['second_name']}",
            'team_name': player['team_name'],
            'team_short': player['team_short'],
            'position': player['position'],
            'position_name': player['position_name'],
            'element_type': position_to_num.get(player['position_name'], 0),
            'now_cost': player['now_cost'],
            'cost_million': player['now_cost'] / 10.0,
            'total_points': player['total_points'],
            'selected_by_percent': player['selected_by_percent'],
            'last_season_total': last_season_total,
            'last_season_ppg': last_season_total / 38,
            'predicted_4gw': predicted_4gw,
            'predicted_ppg': predicted_4gw / 4.0,
            'value_per_million': predicted_4gw / (player['now_cost'] / 10.0),
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by predicted points
    results_df = results_df.sort_values('predicted_4gw', ascending=False)
    
    print(f"✅ Predictions completed for {len(results_df)} players")
    
    # Analysis
    print("\n" + "="*80)
    print("🔮 FANTASY FOOTBALL 2025 PREDICTIONS (4 GAMEWEEKS)")
    print("="*80)
    
    print(f"\nPrediction Statistics:")
    print(f"Mean predicted (4GW): {results_df['predicted_4gw'].mean():.2f}")
    print(f"Max predicted (4GW): {results_df['predicted_4gw'].max():.2f}")
    print(f"Min predicted (4GW): {results_df['predicted_4gw'].min():.2f}")
    print(f"Std predicted (4GW): {results_df['predicted_4gw'].std():.2f}")
    print(f"Players with >25 pts: {(results_df['predicted_4gw'] > 25).sum()}")
    print(f"Players with >20 pts: {(results_df['predicted_4gw'] > 20).sum()}")
    
    # Top performers
    print(f"\n🏆 TOP 15 PREDICTED PERFORMERS (4 Gameweeks):")
    print("-" * 110)
    print(f"{'Rank':<4} {'Player':<25} {'Team':<4} {'Pos':<4} {'Cost':<5} {'4GW':<6} {'PPG':<5} {'LastPPG':<7} {'LastTotal':<9} {'Value':<6}")
    print("-" * 110)
    
    for i, (_, player) in enumerate(results_df.head(15).iterrows(), 1):
        pos_short = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}.get(player['position_name'], 'UNK')
        print(f"{i:<4} {player['player_name'][:24]:<25} {player['team_short']:<4} {pos_short:<4} "
              f"£{player['cost_million']:<4.1f} {player['predicted_4gw']:<6.1f} {player['predicted_ppg']:<5.2f} "
              f"{player['last_season_ppg']:<7.2f} {player['last_season_total']:<9.0f} {player['value_per_million']:<6.2f}")
    
    # Position analysis
    print(f"\n📊 PREDICTIONS BY POSITION:")
    print("-" * 70)
    for pos_name in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_data = results_df[results_df['position_name'] == pos_name]
        if len(pos_data) > 0:
            print(f"{pos_name:12}: Avg {pos_data['predicted_4gw'].mean():5.2f} (4GW), "
                  f"Max {pos_data['predicted_4gw'].max():5.2f}, "
                  f"Players {len(pos_data):3d}")
    
    # Best value picks by position
    print(f"\n💎 BEST VALUE BY POSITION (Points per £M):")
    print("-" * 80)
    for pos_name in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_data = results_df[results_df['position_name'] == pos_name]
        if len(pos_data) > 0:
            best_value = pos_data.sort_values('value_per_million', ascending=False).iloc[0]
            pos_short = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}[pos_name]
            print(f"{pos_short}: {best_value['player_name'][:25]:<25} "
                  f"£{best_value['cost_million']:.1f} - {best_value['predicted_4gw']:.1f} pts - "
                  f"{best_value['value_per_million']:.2f}/£M")
    
    # Team analysis
    print(f"\n🏟️  TOP 10 TEAMS BY AVERAGE PREDICTED POINTS:")
    print("-" * 60)
    team_summary = results_df.groupby('team_short').agg({
        'predicted_4gw': ['mean', 'count'],
        'last_season_total': 'mean'
    }).round(2)
    
    team_summary.columns = ['avg_predicted_4gw', 'player_count', 'avg_last_season']
    team_summary = team_summary.sort_values('avg_predicted_4gw', ascending=False)
    
    for i, (team, data) in enumerate(team_summary.head(10).iterrows(), 1):
        print(f"{i:2d}. {team}: {data['avg_predicted_4gw']:.1f} predicted (4GW) "
              f"({data['avg_last_season']:.0f} avg last season) - {data['player_count']:.0f} players")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"data_2025/predictions/fantasy_predictions_2025_final_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    print(f"\n💾 Results saved to: {filename}")
    print(f"🎯 Ready for 2025 Fantasy Football season!")
    
    # Show Salah specifically
    salah_data = results_df[results_df['second_name'].str.contains('Salah', case=False, na=False)]
    if len(salah_data) > 0:
        salah = salah_data.iloc[0]
        salah_rank = results_df.index[results_df['id'] == salah['id']].tolist()[0] + 1
        print(f"\n⭐ MOHAMED SALAH:")
        print(f"   Rank: #{salah_rank}")
        print(f"   Predicted (4GW): {salah['predicted_4gw']:.2f} points")
        print(f"   Predicted (PPG): {salah['predicted_ppg']:.2f} points")
        print(f"   Last season: {salah['last_season_total']:.0f} points ({salah['last_season_ppg']:.2f} PPG)")
        print(f"   Value: {salah['value_per_million']:.2f} points per £M")

if __name__ == "__main__":
    main()