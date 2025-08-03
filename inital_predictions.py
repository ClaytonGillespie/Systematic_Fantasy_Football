#%%
# Manual injury tracking
def get_injured_players():
    """
    Manually specify injured players - update this list as needed
    Returns dict with player_id as key and injury info as value
    """
    # Update this dictionary with current injury information
    injured_players = {
        # Example format:
        # 123: {'status': 'injured', 'expected_return': 'GW3', 'severity': 'major'},
        # 456: {'status': 'doubt', 'expected_return': 'GW1', 'severity': 'minor'},
        
        # Add injured player IDs here:
        # 1: {'status': 'injured', 'expected_return': 'GW2', 'severity': 'major'},
        # 15: {'status': 'doubt', 'expected_return': 'GW1', 'severity': 'minor'},
    }
    
    return injured_players

def apply_injury_adjustments(elements_df, baseline_df, injured_players):
    """
    Apply injury-based adjustments to baseline stats
    """
    adjusted_baseline = baseline_df.copy()
    
    injury_count = 0
    
    for player_id, injury_info in injured_players.items():
        if player_id in baseline_df['element'].values:
            injury_status = injury_info.get('status', 'injured')
            severity = injury_info.get('severity', 'major')
            
            # Get player info for logging
            player_info = elements_df[elements_df['id'] == player_id]
            if len(player_info) > 0:
                player_name = f"{player_info.iloc[0]['first_name']} {player_info.iloc[0]['second_name']}"
                print(f"⚕️  Injury adjustment: {player_name} - {injury_status} ({severity})")
            
            # Apply adjustments based on injury status
            player_mask = adjusted_baseline['element'] == player_id
            
            if injury_status == 'injured':
                if severity == 'major':
                    # Major injury - severely reduce expectations
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.1
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.1
                else:
                    # Minor injury but still out
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.3
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.3
                    
            elif injury_status == 'doubt':
                if severity == 'major':
                    # Major doubt - significant reduction
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.6
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.7
                else:
                    # Minor doubt - small reduction
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.8
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.9
            
            # Reduce all performance stats for injured players
            performance_cols = ['goals_scored_last5', 'assists_last5', 'bonus_last5', 
                              'bps_last5', 'ict_index_last5', 'expected_goals_last5', 
                              'expected_assists_last5']
            
            for col in performance_cols:
                if col in adjusted_baseline.columns:
                    if injury_status == 'injured':
                        multiplier = 0.1 if severity == 'major' else 0.3
                    else:  # doubt
                        multiplier = 0.6 if severity == 'major' else 0.8
                    
                    adjusted_baseline.loc[player_mask, col] *= multiplier
            
            injury_count += 1
    
    if injury_count > 0:
        print(f"Applied injury adjustments to {injury_count} players")
    else:
        print("No injury adjustments applied")
    
    return adjusted_baseline

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

#%%
# Load your trained model
def load_trained_model(model_path='fantasy_football_future_xgb_gw4.json'):
    """Load the trained XGBoost model"""
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please check the path.")
        return None

#%%
# Load current player data and fixtures
def load_current_data(gameweek=0):
    """Load current season data for predictions"""
    
    # Load current player elements data
    elements_df = pd.read_csv(f'./data_2025/players_gw{gameweek}.csv')
    print(f"Loaded {len(elements_df)} players from players data")
    
    # Load teams data for mapping
    teams_df = pd.read_csv(f'./data_2025/teams_gw{gameweek}.csv')
    print(f"Loaded {len(teams_df)} teams from teams data")
    
    # Load player-specific upcoming fixtures
    player_fixtures_df = pd.read_csv(f'./data_2025/player_upcoming_fixtures_gw{gameweek}.csv')
    print(f"Loaded {len(player_fixtures_df)} player fixtures")
    
    # For start of season, we don't need historical match data
    # We'll use baseline stats instead
    print("Using baseline stats for start of season predictions")
    
    # Load upcoming fixtures (general)
    fixtures_df = pd.read_csv(f'./data_2025/fixtures_gw{gameweek}.csv')
    print(f"Loaded {len(fixtures_df)} future fixtures")
    
    return elements_df, None, fixtures_df, teams_df, player_fixtures_df

#%%
# Calculate baseline stats for each player (pre-season)
def calculate_baseline_stats(elements_df, use_previous_season=True):
    """Calculate baseline stats for players at start of season"""
    
    baseline_stats = []
    
    # Position-based defaults (based on typical performance by position)
    position_defaults = {
        'Goalkeeper': {  # Changed from number to string
            'total_points_baseline': 4.0,
            'minutes_baseline': 85.0,
            'goals_scored_baseline': 0.0,
            'assists_baseline': 0.1,
            'clean_sheets_baseline': 0.3,
            'goals_conceded_baseline': 1.2,
            'saves_baseline': 3.0,
            'bonus_baseline': 0.3,
            'bps_baseline': 25.0,
            'ict_index_baseline': 40.0,
            'expected_goals_baseline': 0.0,
            'expected_assists_baseline': 0.05,
        },
        'Defender': {  # Changed from number to string
            'total_points_baseline': 4.5,
            'minutes_baseline': 80.0,
            'goals_scored_baseline': 0.1,
            'assists_baseline': 0.2,
            'clean_sheets_baseline': 0.3,
            'goals_conceded_baseline': 1.2,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.4,
            'bps_baseline': 30.0,
            'ict_index_baseline': 60.0,
            'expected_goals_baseline': 0.15,
            'expected_assists_baseline': 0.2,
        },
        'Midfielder': {  # Changed from number to string
            'total_points_baseline': 4.0,
            'minutes_baseline': 75.0,
            'goals_scored_baseline': 0.2,
            'assists_baseline': 0.3,
            'clean_sheets_baseline': 0.2,
            'goals_conceded_baseline': 1.3,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.5,
            'bps_baseline': 35.0,
            'ict_index_baseline': 80.0,
            'expected_goals_baseline': 0.25,
            'expected_assists_baseline': 0.3,
        },
        'Forward': {  # Changed from number to string
            'total_points_baseline': 4.2,
            'minutes_baseline': 70.0,
            'goals_scored_baseline': 0.5,
            'assists_baseline': 0.2,
            'clean_sheets_baseline': 0.0,
            'goals_conceded_baseline': 0.0,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.6,
            'bps_baseline': 30.0,
            'ict_index_baseline': 70.0,
            'expected_goals_baseline': 0.6,
            'expected_assists_baseline': 0.15,
        }
    }
    
    for _, player in elements_df.iterrows():
        element_id = player['id']
        position = player['position']  # Now using string position
        
        # Get position defaults
        defaults = position_defaults.get(position, position_defaults['Midfielder'])  # Default to midfielder
        
        # Adjust based on player's price (higher price = better expected performance)
        price_multiplier = min(max(player['now_cost'] / 75.0, 0.5), 2.0)  # Scale between 0.5x and 2x
        
        # Adjust based on player's selection percentage (higher ownership = better expected)
        selection_multiplier = min(max(player['selected_by_percent'] / 10.0, 0.7), 1.5)
        
        # Use last season's total points if available and sensible
        if pd.notna(player['total_points']) and player['total_points'] > 0:
            # Use last season's average but cap it reasonably
            last_season_avg = min(player['total_points'] / 38.0, 10.0)  # Cap at 10 points per game
            baseline_points = (last_season_avg * 0.7 + defaults['total_points_baseline'] * 0.3)
        else:
            baseline_points = defaults['total_points_baseline']
        
        # Apply multipliers
        baseline_points *= price_multiplier * selection_multiplier
        
        baseline_metrics = {
            'element': element_id,
            'total_points_last5': baseline_points,
            'minutes_last5': defaults['minutes_baseline'] * price_multiplier,
            'goals_scored_last5': defaults['goals_scored_baseline'] * price_multiplier,
            'assists_last5': defaults['assists_baseline'] * price_multiplier,
            'clean_sheets_last5': defaults['clean_sheets_baseline'],
            'goals_conceded_last5': defaults['goals_conceded_baseline'],
            'bonus_last5': defaults['bonus_baseline'] * price_multiplier,
            'bps_last5': defaults['bps_baseline'] * price_multiplier,
            'ict_index_last5': defaults['ict_index_baseline'] * price_multiplier,
            'expected_goals_last5': defaults['expected_goals_baseline'] * price_multiplier,
            'expected_assists_last5': defaults['expected_assists_baseline'] * price_multiplier,
            'saves_last5': defaults['saves_baseline'] * price_multiplier if position == 1 else 0.0,
            'yellow_cards_last5': 0.1,  # Low baseline for cards
            'red_cards_last5': 0.01,
            'starts_last5': min(defaults['minutes_baseline'] / 90.0, 1.0),
            'influence_last5': defaults['ict_index_baseline'] * 0.4 * price_multiplier,
            'creativity_last5': defaults['ict_index_baseline'] * 0.3 * price_multiplier,
            'threat_last5': defaults['ict_index_baseline'] * 0.3 * price_multiplier,
        }
        
        baseline_stats.append(baseline_metrics)
    
    baseline_df = pd.DataFrame(baseline_stats)
    print(f"Created baseline stats for {len(baseline_df)} players")
    
    # Show sample baseline stats by position
    print("\nBaseline Stats Summary by Position:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_players = elements_df[elements_df['position'] == pos]
        if len(pos_players) > 0:
            pos_baseline = baseline_df[baseline_df['element'].isin(pos_players['id'])]
            avg_points = pos_baseline['total_points_last5'].mean()
            avg_goals = pos_baseline['goals_scored_last5'].mean()
            avg_assists = pos_baseline['assists_last5'].mean()
            print(f"  {pos}: Avg Points: {avg_points:.2f}, Goals: {avg_goals:.2f}, Assists: {avg_assists:.2f}")
    
    return baseline_df

#%%
# Get upcoming fixture difficulty for each player using player-specific data
def get_upcoming_fixtures_from_player_data(elements_df, player_fixtures_df, target_gameweeks=4):
    """Get upcoming fixture difficulty for each player using player-specific fixture data"""
    
    fixture_stats = []
    
    # Get current gameweek (approximate based on latest fixtures)
    current_gw = player_fixtures_df['event'].min()
    upcoming_gws = list(range(current_gw, current_gw + target_gameweeks))
    
    print(f"Analyzing player fixtures for gameweeks {upcoming_gws}")
    
    for _, player in elements_df.iterrows():
        player_id = player['id']
        
        # Find fixtures for this specific player
        player_upcoming = player_fixtures_df[
            (player_fixtures_df['player_id'] == player_id) &
            (player_fixtures_df['event'].isin(upcoming_gws)) &
            (player_fixtures_df['finished'] == False)
        ]
        
        if len(player_upcoming) > 0:
            # Calculate fixture difficulty and home ratio
            difficulties = player_upcoming['difficulty'].tolist()
            home_count = player_upcoming['is_home'].sum()
            
            avg_difficulty = np.mean(difficulties) if difficulties else 3.0
            home_ratio = home_count / len(player_upcoming)
            
            fixture_stats.append({
                'element': player_id,
                'next_fixture_difficulty': avg_difficulty,
                'next_home_ratio': home_ratio,
                'upcoming_fixtures': len(player_upcoming)
            })
        else:
            # No fixtures found for this player
            fixture_stats.append({
                'element': player_id,
                'next_fixture_difficulty': 3.0,
                'next_home_ratio': 0.5,
                'upcoming_fixtures': 0
            })
    
    return pd.DataFrame(fixture_stats)

#%%
# Create prediction features for all players
def create_prediction_features(elements_df, baseline_df, fixtures_df):
    """Create feature matrix for predictions using baseline stats"""
    
    # Merge all data
    prediction_data = elements_df.merge(baseline_df, left_on='id', right_on='element', how='left')
    prediction_data = prediction_data.merge(fixtures_df, on='element', how='left')
    
    # Only fill NaNs for critical calculated features that would break operations
    # Let XGBoost handle naturally missing features
    critical_fills = {
        'next_fixture_difficulty': 3.0,  # Default difficulty
        'next_home_ratio': 0.5,          # 50/50 home/away
        'upcoming_fixtures': 4,          # Assume full gameweeks
        # Add defaults for missing columns that are required by the model
        'transfers_in': 100,
        'transfers_out': 100,
        'transfers_balance': 0,
        'selected': 1000,
        'value': 50
    }
    
    prediction_data = prediction_data.fillna(critical_fills)
    
    # Create additional features with available data
    current_time = datetime.now()
    prediction_data['hour'] = 15  # Typical kickoff time
    prediction_data['day_of_week'] = 5  # Saturday
    prediction_data['month'] = current_time.month
    prediction_data['round'] = 1  # Start of season
    prediction_data['season'] = 2025
    prediction_data['was_home'] = (prediction_data['next_home_ratio'] > 0.5).astype(int)
    prediction_data['opponent_team'] = 10  # Average opponent
    
    # Use baseline stats for game performance metrics
    prediction_data['minutes'] = prediction_data['minutes_last5']
    prediction_data['goals_scored'] = prediction_data['goals_scored_last5'] 
    prediction_data['assists'] = prediction_data['assists_last5']
    prediction_data['clean_sheets'] = prediction_data['clean_sheets_last5']
    prediction_data['goals_conceded'] = prediction_data['goals_conceded_last5']
    prediction_data['own_goals'] = 0
    prediction_data['penalties_saved'] = 0
    prediction_data['penalties_missed'] = 0
    prediction_data['yellow_cards'] = prediction_data['yellow_cards_last5']
    prediction_data['red_cards'] = prediction_data['red_cards_last5']
    prediction_data['saves'] = prediction_data['saves_last5']
    prediction_data['bonus'] = prediction_data['bonus_last5']
    prediction_data['bps'] = prediction_data['bps_last5']
    prediction_data['influence'] = prediction_data['influence_last5']
    prediction_data['creativity'] = prediction_data['creativity_last5']
    prediction_data['threat'] = prediction_data['threat_last5']
    prediction_data['ict_index'] = prediction_data['ict_index_last5']
    prediction_data['starts'] = prediction_data['starts_last5']
    prediction_data['expected_goals'] = prediction_data['expected_goals_last5']
    prediction_data['expected_assists'] = prediction_data['expected_assists_last5']
    prediction_data['expected_goal_involvements'] = prediction_data['expected_goals'] + prediction_data['expected_assists']
    prediction_data['expected_goals_conceded'] = prediction_data['goals_conceded_last5']
    
    # Market data (use available columns, set defaults for missing ones)
    prediction_data['value'] = prediction_data['now_cost']
    prediction_data['transfers_balance'] = 0  # Not available in new format
    prediction_data['selected'] = prediction_data['selected_by_percent'] * 1000  # Approximate
    prediction_data['transfers_in'] = 100  # Default value
    prediction_data['transfers_out'] = 100  # Default value
    
    # Rolling features (use last 5 as last 3 approximation)
    prediction_data['total_points_last3'] = prediction_data['total_points_last5']
    prediction_data['minutes_last3'] = prediction_data['minutes_last5']
    prediction_data['goals_scored_last3'] = prediction_data['goals_scored_last5']
    prediction_data['assists_last3'] = prediction_data['assists_last5']
    prediction_data['expected_goals_last3'] = prediction_data['expected_goals_last5']
    prediction_data['expected_assists_last3'] = prediction_data['expected_assists_last5']
    prediction_data['ict_index_last3'] = prediction_data['ict_index_last5']
    prediction_data['bps_last3'] = prediction_data['bps_last5']
    
    # Form indicators
    prediction_data['recent_form'] = prediction_data['total_points_last5']
    prediction_data['goals_form'] = prediction_data['goals_scored_last5']
    prediction_data['assists_form'] = prediction_data['assists_last5']
    
    # Team strength (simplified)
    prediction_data['team_goals_scored'] = prediction_data['goals_scored_last5'] * 11  # Team approximation
    prediction_data['team_goals_conceded'] = prediction_data['goals_conceded_last5'] * 11
    
    return prediction_data

#%%
# Generate expected scores for all players
def generate_expected_scores(model, prediction_data, target_gameweeks=4, gameweek=0):
    """Generate expected scores using the trained model"""
    
    # Define feature columns (same as training)
    feature_columns = [
        'opponent_team', 'was_home', 'round', 'season',
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
        'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'starts',
        'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
        'value', 'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
        'hour', 'day_of_week', 'month',
        'total_points_last3', 'minutes_last3', 'goals_scored_last3', 'assists_last3',
        'expected_goals_last3', 'expected_assists_last3', 'ict_index_last3', 'bps_last3',
        'next_fixture_difficulty', 'next_home_ratio',
        'recent_form', 'goals_form', 'assists_form',
        'team_goals_scored', 'team_goals_conceded'
    ]
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in prediction_data.columns]
    print(f"Using {len(available_features)} features for predictions")
    
    # Create feature matrix - let XGBoost handle most missing values
    X = prediction_data[available_features]
    
    # Only fill NaNs that would break operations or weren't in training
    # XGBoost can handle the rest natively
    X = X.fillna({
        'next_fixture_difficulty': 3.0,
        'next_home_ratio': 0.5,
        'upcoming_fixtures': 4,
        'hour': 15,
        'day_of_week': 5,
        'month': 8
    })
    
    # Generate predictions
    expected_scores = model.predict(X)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'element_id': prediction_data['id'],
        'web_name': prediction_data['first_name'] + ' ' + prediction_data['second_name'],
        'first_name': prediction_data['first_name'],
        'second_name': prediction_data['second_name'],
        'team': prediction_data['team_name'],
        'element_type': prediction_data['position'],
        'now_cost': prediction_data['now_cost'],
        'selected_by_percent': prediction_data['selected_by_percent'],
        'form': prediction_data['form'],
        'total_points': prediction_data['total_points'],
        'points_per_game': prediction_data['total_points'] / 38.0,  # Approximate PPG
        f'expected_points_next_{target_gameweeks}gw': expected_scores,
        f'expected_points_per_gw': expected_scores / target_gameweeks,
        'recent_form_last5': prediction_data['total_points_last5'],
        'next_fixture_difficulty': prediction_data['next_fixture_difficulty'],
        'next_home_ratio': prediction_data['next_home_ratio'],
        'upcoming_fixtures': prediction_data['upcoming_fixtures']
    })
    
    # Add injury status to results
    injured_players = get_injured_players()
    results_df['injury_status'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('status', 'available')
    )
    results_df['injury_severity'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('severity', 'none')
    )
    results_df['expected_return'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('expected_return', '')
    )
    
    # Add value metrics
    results_df['cost_per_million'] = results_df['now_cost'] / 10
    results_df[f'expected_points_per_million'] = results_df[f'expected_points_next_{target_gameweeks}gw'] / results_df['cost_per_million']
    results_df['value_score'] = results_df[f'expected_points_per_million'] * 100  # Scale for readability
    
    # Sort by expected points
    results_df = results_df.sort_values(f'expected_points_next_{target_gameweeks}gw', ascending=False)
    
    return results_df

#%%
# Main execution
def main():
    print("🏆 Fantasy Football Expected Scores Generator")
    print("=" * 50)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Load current data
    gameweek = 0  # Change this to predict for different gameweeks
    elements_df, historical_df, fixtures_df, teams_df, player_fixtures_df = load_current_data(gameweek)
    
    # Calculate baseline stats (instead of recent form for start of season)
    print("\nCalculating baseline stats for start of season...")
    baseline_df = calculate_baseline_stats(elements_df)
    
    # Apply injury adjustments
    print("\nApplying injury adjustments...")
    injured_players = get_injured_players()
    baseline_df = apply_injury_adjustments(elements_df, baseline_df, injured_players)
    
    # Get upcoming fixtures using player-specific data
    print("Analyzing upcoming fixtures...")
    fixture_stats = get_upcoming_fixtures_from_player_data(elements_df, player_fixtures_df)
    
    # Create prediction features
    print("Creating prediction features...")
    prediction_data = create_prediction_features(elements_df, baseline_df, fixture_stats)
    
    # Generate expected scores
    print("Generating expected scores...")
    results_df = generate_expected_scores(model, prediction_data, gameweek=gameweek)
    
    # Display top players
    print(f"\n🌟 Top 20 Expected Performers (Next 4 Gameweeks):")
    print("=" * 80)
    top_20 = results_df.head(20)
    
    for idx, player in top_20.iterrows():
        pos = player['element_type']  # Now already a string
        print(f"{player['web_name']:20} ({pos}) "
              f"Expected: {player['expected_points_next_4gw']:5.2f} "
              f"Per GW: {player['expected_points_per_gw']:4.2f} "
              f"Cost: £{player['cost_per_million']:4.1f}m "
              f"Value: {player['value_score']:5.1f}")
    
    # Save results
    output_file = f'./data_2025/player_expected_scores_gw{gameweek}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
#%%
# Manual injury tracking
def get_injured_players():
    """
    Manually specify injured players - update this list as needed
    Returns dict with player_id as key and injury info as value
    """
    # Update this dictionary with current injury information
    injured_players = {
        # Example format:
        # 123: {'status': 'injured', 'expected_return': 'GW3', 'severity': 'major'},
        # 456: {'status': 'doubt', 'expected_return': 'GW1', 'severity': 'minor'},
        
        # Add injured player IDs here:
        # 1: {'status': 'injured', 'expected_return': 'GW2', 'severity': 'major'},
        #Luiz Diaz to Bayern
        383: {'status': 'injured', 'expected_return': 'GW38', 'severity': 'major'},
    }
    
    return injured_players

def apply_injury_adjustments(elements_df, baseline_df, injured_players):
    """
    Apply injury-based adjustments to baseline stats
    """
    adjusted_baseline = baseline_df.copy()
    
    injury_count = 0
    
    for player_id, injury_info in injured_players.items():
        if player_id in baseline_df['element'].values:
            injury_status = injury_info.get('status', 'injured')
            severity = injury_info.get('severity', 'major')
            
            # Get player info for logging
            player_info = elements_df[elements_df['id'] == player_id]
            if len(player_info) > 0:
                player_name = f"{player_info.iloc[0]['first_name']} {player_info.iloc[0]['second_name']}"
                print(f"⚕️  Injury adjustment: {player_name} - {injury_status} ({severity})")
            
            # Apply adjustments based on injury status
            player_mask = adjusted_baseline['element'] == player_id
            
            if injury_status == 'injured':
                if severity == 'major':
                    # Major injury - severely reduce expectations
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.1
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.1
                else:
                    # Minor injury but still out
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.3
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.3
                    
            elif injury_status == 'doubt':
                if severity == 'major':
                    # Major doubt - significant reduction
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.6
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.7
                else:
                    # Minor doubt - small reduction
                    adjusted_baseline.loc[player_mask, 'total_points_last5'] *= 0.8
                    adjusted_baseline.loc[player_mask, 'minutes_last5'] *= 0.9
            
            # Reduce all performance stats for injured players
            performance_cols = ['goals_scored_last5', 'assists_last5', 'bonus_last5', 
                              'bps_last5', 'ict_index_last5', 'expected_goals_last5', 
                              'expected_assists_last5']
            
            for col in performance_cols:
                if col in adjusted_baseline.columns:
                    if injury_status == 'injured':
                        multiplier = 0.1 if severity == 'major' else 0.3
                    else:  # doubt
                        multiplier = 0.6 if severity == 'major' else 0.8
                    
                    adjusted_baseline.loc[player_mask, col] *= multiplier
            
            injury_count += 1
    
    if injury_count > 0:
        print(f"Applied injury adjustments to {injury_count} players")
    else:
        print("No injury adjustments applied")
    
    return adjusted_baseline

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

#%%
# Load your trained model
def load_trained_model(model_path='fantasy_football_future_xgb_gw4.json'):
    """Load the trained XGBoost model"""
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please check the path.")
        return None

#%%
# Load current player data and fixtures
def load_current_data(gameweek=0):
    """Load current season data for predictions"""
    
    # Load current player elements data
    elements_df = pd.read_csv(f'./data_2025/players_gw{gameweek}.csv')
    print(f"Loaded {len(elements_df)} players from players data")
    
    # Load teams data for mapping
    teams_df = pd.read_csv(f'./data_2025/teams_gw{gameweek}.csv')
    print(f"Loaded {len(teams_df)} teams from teams data")
    
    # Load player-specific upcoming fixtures
    player_fixtures_df = pd.read_csv(f'./data_2025/player_upcoming_fixtures_gw{gameweek}.csv')
    print(f"Loaded {len(player_fixtures_df)} player fixtures")
    
    # For start of season, we don't need historical match data
    # We'll use baseline stats instead
    print("Using baseline stats for start of season predictions")
    
    # Load upcoming fixtures (general)
    fixtures_df = pd.read_csv(f'./data_2025/fixtures_gw{gameweek}.csv')
    print(f"Loaded {len(fixtures_df)} future fixtures")
    
    return elements_df, None, fixtures_df, teams_df, player_fixtures_df

#%%
# Calculate baseline stats for each player (pre-season)
def calculate_baseline_stats(elements_df, use_previous_season=True):
    """Calculate baseline stats for players at start of season"""
    
    baseline_stats = []
    
    # Position-based defaults (based on typical performance by position)
    position_defaults = {
        'Goalkeeper': {  # Changed from number to string
            'total_points_baseline': 4.0,
            'minutes_baseline': 85.0,
            'goals_scored_baseline': 0.0,
            'assists_baseline': 0.1,
            'clean_sheets_baseline': 0.3,
            'goals_conceded_baseline': 1.2,
            'saves_baseline': 3.0,
            'bonus_baseline': 0.3,
            'bps_baseline': 25.0,
            'ict_index_baseline': 40.0,
            'expected_goals_baseline': 0.0,
            'expected_assists_baseline': 0.05,
        },
        'Defender': {  # Changed from number to string
            'total_points_baseline': 4.5,
            'minutes_baseline': 80.0,
            'goals_scored_baseline': 0.1,
            'assists_baseline': 0.2,
            'clean_sheets_baseline': 0.3,
            'goals_conceded_baseline': 1.2,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.4,
            'bps_baseline': 30.0,
            'ict_index_baseline': 60.0,
            'expected_goals_baseline': 0.15,
            'expected_assists_baseline': 0.2,
        },
        'Midfielder': {  # Changed from number to string
            'total_points_baseline': 4.0,
            'minutes_baseline': 75.0,
            'goals_scored_baseline': 0.2,
            'assists_baseline': 0.3,
            'clean_sheets_baseline': 0.2,
            'goals_conceded_baseline': 1.3,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.5,
            'bps_baseline': 35.0,
            'ict_index_baseline': 80.0,
            'expected_goals_baseline': 0.25,
            'expected_assists_baseline': 0.3,
        },
        'Forward': {  # Changed from number to string
            'total_points_baseline': 4.2,
            'minutes_baseline': 70.0,
            'goals_scored_baseline': 0.5,
            'assists_baseline': 0.2,
            'clean_sheets_baseline': 0.0,
            'goals_conceded_baseline': 0.0,
            'saves_baseline': 0.0,
            'bonus_baseline': 0.6,
            'bps_baseline': 30.0,
            'ict_index_baseline': 70.0,
            'expected_goals_baseline': 0.6,
            'expected_assists_baseline': 0.15,
        }
    }
    
    for _, player in elements_df.iterrows():
        element_id = player['id']
        position = player['position']  # Now using string position
        
        # Get position defaults
        defaults = position_defaults.get(position, position_defaults['Midfielder'])  # Default to midfielder
        
        # Adjust based on player's price (higher price = better expected performance)
        price_multiplier = min(max(player['now_cost'] / 75.0, 0.5), 2.0)  # Scale between 0.5x and 2x
        
        # Adjust based on player's selection percentage (higher ownership = better expected)
        selection_multiplier = min(max(player['selected_by_percent'] / 10.0, 0.7), 1.5)
        
        # Use last season's total points if available and sensible
        if pd.notna(player['total_points']) and player['total_points'] > 0:
            # Use last season's average but cap it reasonably
            last_season_avg = min(player['total_points'] / 38.0, 10.0)  # Cap at 10 points per game
            baseline_points = (last_season_avg * 0.7 + defaults['total_points_baseline'] * 0.3)
        else:
            baseline_points = defaults['total_points_baseline']
        
        # Apply multipliers
        baseline_points *= price_multiplier * selection_multiplier
        
        baseline_metrics = {
            'element': element_id,
            'total_points_last5': baseline_points,
            'minutes_last5': defaults['minutes_baseline'] * price_multiplier,
            'goals_scored_last5': defaults['goals_scored_baseline'] * price_multiplier,
            'assists_last5': defaults['assists_baseline'] * price_multiplier,
            'clean_sheets_last5': defaults['clean_sheets_baseline'],
            'goals_conceded_last5': defaults['goals_conceded_baseline'],
            'bonus_last5': defaults['bonus_baseline'] * price_multiplier,
            'bps_last5': defaults['bps_baseline'] * price_multiplier,
            'ict_index_last5': defaults['ict_index_baseline'] * price_multiplier,
            'expected_goals_last5': defaults['expected_goals_baseline'] * price_multiplier,
            'expected_assists_last5': defaults['expected_assists_baseline'] * price_multiplier,
            'saves_last5': defaults['saves_baseline'] * price_multiplier if position == 1 else 0.0,
            'yellow_cards_last5': 0.1,  # Low baseline for cards
            'red_cards_last5': 0.01,
            'starts_last5': min(defaults['minutes_baseline'] / 90.0, 1.0),
            'influence_last5': defaults['ict_index_baseline'] * 0.4 * price_multiplier,
            'creativity_last5': defaults['ict_index_baseline'] * 0.3 * price_multiplier,
            'threat_last5': defaults['ict_index_baseline'] * 0.3 * price_multiplier,
        }
        
        baseline_stats.append(baseline_metrics)
    
    baseline_df = pd.DataFrame(baseline_stats)
    print(f"Created baseline stats for {len(baseline_df)} players")
    
    # Show sample baseline stats by position
    print("\nBaseline Stats Summary by Position:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_players = elements_df[elements_df['position'] == pos]
        if len(pos_players) > 0:
            pos_baseline = baseline_df[baseline_df['element'].isin(pos_players['id'])]
            avg_points = pos_baseline['total_points_last5'].mean()
            avg_goals = pos_baseline['goals_scored_last5'].mean()
            avg_assists = pos_baseline['assists_last5'].mean()
            print(f"  {pos}: Avg Points: {avg_points:.2f}, Goals: {avg_goals:.2f}, Assists: {avg_assists:.2f}")
    
    return baseline_df

#%%
# Get upcoming fixture difficulty for each player using player-specific data
def get_upcoming_fixtures_from_player_data(elements_df, player_fixtures_df, target_gameweeks=4):
    """Get upcoming fixture difficulty for each player using player-specific fixture data"""
    
    fixture_stats = []
    
    # Get current gameweek (approximate based on latest fixtures)
    current_gw = player_fixtures_df['event'].min()
    upcoming_gws = list(range(current_gw, current_gw + target_gameweeks))
    
    print(f"Analyzing player fixtures for gameweeks {upcoming_gws}")
    
    for _, player in elements_df.iterrows():
        player_id = player['id']
        
        # Find fixtures for this specific player
        player_upcoming = player_fixtures_df[
            (player_fixtures_df['player_id'] == player_id) &
            (player_fixtures_df['event'].isin(upcoming_gws)) &
            (player_fixtures_df['finished'] == False)
        ]
        
        if len(player_upcoming) > 0:
            # Calculate fixture difficulty and home ratio
            difficulties = player_upcoming['difficulty'].tolist()
            home_count = player_upcoming['is_home'].sum()
            
            avg_difficulty = np.mean(difficulties) if difficulties else 3.0
            home_ratio = home_count / len(player_upcoming)
            
            fixture_stats.append({
                'element': player_id,
                'next_fixture_difficulty': avg_difficulty,
                'next_home_ratio': home_ratio,
                'upcoming_fixtures': len(player_upcoming)
            })
        else:
            # No fixtures found for this player
            fixture_stats.append({
                'element': player_id,
                'next_fixture_difficulty': 3.0,
                'next_home_ratio': 0.5,
                'upcoming_fixtures': 0
            })
    
    return pd.DataFrame(fixture_stats)

#%%
# Create prediction features for all players
def create_prediction_features(elements_df, baseline_df, fixtures_df):
    """Create feature matrix for predictions using baseline stats"""
    
    # Merge all data
    prediction_data = elements_df.merge(baseline_df, left_on='id', right_on='element', how='left')
    prediction_data = prediction_data.merge(fixtures_df, on='element', how='left')
    
    # Only fill NaNs for critical calculated features that would break operations
    # Let XGBoost handle naturally missing features
    critical_fills = {
        'next_fixture_difficulty': 3.0,  # Default difficulty
        'next_home_ratio': 0.5,          # 50/50 home/away
        'upcoming_fixtures': 4,          # Assume full gameweeks
        # Add defaults for missing columns that are required by the model
        'transfers_in': 100,
        'transfers_out': 100,
        'transfers_balance': 0,
        'selected': 1000,
        'value': 50
    }
    
    prediction_data = prediction_data.fillna(critical_fills)
    
    # Create additional features with available data
    current_time = datetime.now()
    prediction_data['hour'] = 15  # Typical kickoff time
    prediction_data['day_of_week'] = 5  # Saturday
    prediction_data['month'] = current_time.month
    prediction_data['round'] = 1  # Start of season
    prediction_data['season'] = 2025
    prediction_data['was_home'] = (prediction_data['next_home_ratio'] > 0.5).astype(int)
    prediction_data['opponent_team'] = 10  # Average opponent
    
    # Use baseline stats for game performance metrics
    prediction_data['minutes'] = prediction_data['minutes_last5']
    prediction_data['goals_scored'] = prediction_data['goals_scored_last5'] 
    prediction_data['assists'] = prediction_data['assists_last5']
    prediction_data['clean_sheets'] = prediction_data['clean_sheets_last5']
    prediction_data['goals_conceded'] = prediction_data['goals_conceded_last5']
    prediction_data['own_goals'] = 0
    prediction_data['penalties_saved'] = 0
    prediction_data['penalties_missed'] = 0
    prediction_data['yellow_cards'] = prediction_data['yellow_cards_last5']
    prediction_data['red_cards'] = prediction_data['red_cards_last5']
    prediction_data['saves'] = prediction_data['saves_last5']
    prediction_data['bonus'] = prediction_data['bonus_last5']
    prediction_data['bps'] = prediction_data['bps_last5']
    prediction_data['influence'] = prediction_data['influence_last5']
    prediction_data['creativity'] = prediction_data['creativity_last5']
    prediction_data['threat'] = prediction_data['threat_last5']
    prediction_data['ict_index'] = prediction_data['ict_index_last5']
    prediction_data['starts'] = prediction_data['starts_last5']
    prediction_data['expected_goals'] = prediction_data['expected_goals_last5']
    prediction_data['expected_assists'] = prediction_data['expected_assists_last5']
    prediction_data['expected_goal_involvements'] = prediction_data['expected_goals'] + prediction_data['expected_assists']
    prediction_data['expected_goals_conceded'] = prediction_data['goals_conceded_last5']
    
    # Market data (use available columns, set defaults for missing ones)
    prediction_data['value'] = prediction_data['now_cost']
    prediction_data['transfers_balance'] = 0  # Not available in new format
    prediction_data['selected'] = prediction_data['selected_by_percent'] * 1000  # Approximate
    prediction_data['transfers_in'] = 100  # Default value
    prediction_data['transfers_out'] = 100  # Default value
    
    # Rolling features (use last 5 as last 3 approximation)
    prediction_data['total_points_last3'] = prediction_data['total_points_last5']
    prediction_data['minutes_last3'] = prediction_data['minutes_last5']
    prediction_data['goals_scored_last3'] = prediction_data['goals_scored_last5']
    prediction_data['assists_last3'] = prediction_data['assists_last5']
    prediction_data['expected_goals_last3'] = prediction_data['expected_goals_last5']
    prediction_data['expected_assists_last3'] = prediction_data['expected_assists_last5']
    prediction_data['ict_index_last3'] = prediction_data['ict_index_last5']
    prediction_data['bps_last3'] = prediction_data['bps_last5']
    
    # Form indicators
    prediction_data['recent_form'] = prediction_data['total_points_last5']
    prediction_data['goals_form'] = prediction_data['goals_scored_last5']
    prediction_data['assists_form'] = prediction_data['assists_last5']
    
    # Team strength (simplified)
    prediction_data['team_goals_scored'] = prediction_data['goals_scored_last5'] * 11  # Team approximation
    prediction_data['team_goals_conceded'] = prediction_data['goals_conceded_last5'] * 11
    
    return prediction_data

#%%
# Generate expected scores for all players
def generate_expected_scores(model, prediction_data, target_gameweeks=4, gameweek=0):
    """Generate expected scores using the trained model"""
    
    # Define feature columns (same as training)
    feature_columns = [
        'opponent_team', 'was_home', 'round', 'season',
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
        'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'starts',
        'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
        'value', 'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
        'hour', 'day_of_week', 'month',
        'total_points_last3', 'minutes_last3', 'goals_scored_last3', 'assists_last3',
        'expected_goals_last3', 'expected_assists_last3', 'ict_index_last3', 'bps_last3',
        'next_fixture_difficulty', 'next_home_ratio',
        'recent_form', 'goals_form', 'assists_form',
        'team_goals_scored', 'team_goals_conceded'
    ]
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in prediction_data.columns]
    print(f"Using {len(available_features)} features for predictions")
    
    # Create feature matrix - let XGBoost handle most missing values
    X = prediction_data[available_features]
    
    # Only fill NaNs that would break operations or weren't in training
    # XGBoost can handle the rest natively
    X = X.fillna({
        'next_fixture_difficulty': 3.0,
        'next_home_ratio': 0.5,
        'upcoming_fixtures': 4,
        'hour': 15,
        'day_of_week': 5,
        'month': 8
    })
    
    # Generate predictions
    expected_scores = model.predict(X)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'element_id': prediction_data['id'],
        'web_name': prediction_data['first_name'] + ' ' + prediction_data['second_name'],
        'first_name': prediction_data['first_name'],
        'second_name': prediction_data['second_name'],
        'team': prediction_data['team_name'],
        'element_type': prediction_data['position'],
        'now_cost': prediction_data['now_cost'],
        'selected_by_percent': prediction_data['selected_by_percent'],
        'form': prediction_data['form'],
        'total_points': prediction_data['total_points'],
        'points_per_game': prediction_data['total_points'] / 38.0,  # Approximate PPG
        f'expected_points_next_{target_gameweeks}gw': expected_scores,
        f'expected_points_per_gw': expected_scores / target_gameweeks,
        'recent_form_last5': prediction_data['total_points_last5'],
        'next_fixture_difficulty': prediction_data['next_fixture_difficulty'],
        'next_home_ratio': prediction_data['next_home_ratio'],
        'upcoming_fixtures': prediction_data['upcoming_fixtures']
    })
    
    # Add injury status to results
    injured_players = get_injured_players()
    results_df['injury_status'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('status', 'available')
    )
    results_df['injury_severity'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('severity', 'none')
    )
    results_df['expected_return'] = results_df['element_id'].apply(
        lambda x: injured_players.get(x, {}).get('expected_return', '')
    )
    
    # Add value metrics
    results_df['cost_per_million'] = results_df['now_cost'] / 10
    results_df[f'expected_points_per_million'] = results_df[f'expected_points_next_{target_gameweeks}gw'] / results_df['cost_per_million']
    results_df['value_score'] = results_df[f'expected_points_per_million'] * 100  # Scale for readability
    
    # Sort by expected points
    results_df = results_df.sort_values(f'expected_points_next_{target_gameweeks}gw', ascending=False)
    
    return results_df

#%%
# Main execution
def main():
    print("🏆 Fantasy Football Expected Scores Generator")
    print("=" * 50)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Load current data
    gameweek = 0  # Change this to predict for different gameweeks
    elements_df, historical_df, fixtures_df, teams_df, player_fixtures_df = load_current_data(gameweek)
    
    # Calculate baseline stats (instead of recent form for start of season)
    print("\nCalculating baseline stats for start of season...")
    baseline_df = calculate_baseline_stats(elements_df)
    
    # Apply injury adjustments
    print("\nApplying injury adjustments...")
    injured_players = get_injured_players()
    baseline_df = apply_injury_adjustments(elements_df, baseline_df, injured_players)
    
    # Get upcoming fixtures using player-specific data
    print("Analyzing upcoming fixtures...")
    fixture_stats = get_upcoming_fixtures_from_player_data(elements_df, player_fixtures_df)
    
    # Create prediction features
    print("Creating prediction features...")
    prediction_data = create_prediction_features(elements_df, baseline_df, fixture_stats)
    
    # Generate expected scores
    print("Generating expected scores...")
    results_df = generate_expected_scores(model, prediction_data, gameweek=gameweek)
    
    # Display top players
    print(f"\n🌟 Top 20 Expected Performers (Next 4 Gameweeks):")
    print("=" * 80)
    top_20 = results_df.head(20)
    
    for idx, player in top_20.iterrows():
        pos = player['element_type']  # Now already a string
        print(f"{player['web_name']:20} ({pos}) "
              f"Expected: {player['expected_points_next_4gw']:5.2f} "
              f"Per GW: {player['expected_points_per_gw']:4.2f} "
              f"Cost: £{player['cost_per_million']:4.1f}m "
              f"Value: {player['value_score']:5.1f}")
    
    # Save results
    output_file = f'./data_2025/player_expected_scores_gw{gameweek}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    # Position-wise analysis
    # Position-wise analysis
    print(f"\n📊 By Position (Top 5 each):")
    print("=" * 50)

    for pos_name in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_players = results_df[results_df['element_type'] == pos_name].head(5)
        if len(pos_players) > 0:
            print(f"\n{pos_name}:")
            for _, player in pos_players.iterrows():
                print(f"  {player['web_name']:20} Expected: {player['expected_points_next_4gw']:5.2f} "
                    f"Value: {player['value_score']:5.1f}")

# Run the analysis
if __name__ == "__main__":
    expected_scores_df = main()
#%%