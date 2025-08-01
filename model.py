
#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
# Load multiple seasons of data with player name matching
def load_multi_season_data():
    """
    Load and combine data from multiple seasons with proper player matching
    """
    seasons_data = []
    
    # Load player names/IDs (this helps map players across seasons)
    try:
        elements_df = pd.read_csv('./data_2024/elements_1.csv')
        print(f"Loaded player elements data: {elements_df.shape[0]} players")
        
        # Create player name to ID mapping
        elements_df['full_name'] = elements_df['first_name'].str.strip() + '_' + elements_df['second_name'].str.strip()
        player_name_map = dict(zip(elements_df['full_name'], elements_df['id']))
        
        print(f"Sample player names: {list(elements_df['full_name'].head())}")
        
    except FileNotFoundError:
        print("Player elements file not found - using element IDs directly")
        elements_df = None
        player_name_map = {}
    
    # Load 2024 data
    try:
        hist_2024 = pd.read_csv('./data_2024/previous_fixtures_38.csv')
        fixtures_2024 = pd.read_csv('./data_2024/future_fixtures_5.csv')
        hist_2024['season'] = 2024
        fixtures_2024['season'] = 2024
        seasons_data.append((hist_2024, fixtures_2024))
        print(f"Loaded 2024 data: {hist_2024.shape[0]} historical records")
    except FileNotFoundError:
        print("2024 data not found")
    
    # Load 2023 data (if available)
    try:
        hist_2023 = pd.read_csv('./data_2023/previous_fixtures_38.csv')
        fixtures_2023 = pd.read_csv('./data_2023/fixtures_38.csv')
        hist_2023['season'] = 2023
        fixtures_2023['season'] = 2023
        
        # Standardize fixtures_2023 to match future_fixtures format
        if 'difficulty' not in fixtures_2023.columns:
            fixtures_2023['difficulty'] = (fixtures_2023['team_h_difficulty'] + fixtures_2023['team_a_difficulty']) / 2
        
        # If we have 2023 elements data, map old IDs to new IDs using names
        if elements_df is not None:
            try:
                elements_2023 = pd.read_csv('./data_2023/elements_1.csv')
                elements_2023['full_name'] = elements_2023['first_name'].str.strip() + '_' + elements_2023['second_name'].str.strip()
                
                # Create mapping from 2023 ID to 2024 ID via names
                id_mapping_2023_to_2024 = {}
                for _, row in elements_2023.iterrows():
                    name = row['full_name']
                    old_id = row['id']
                    if name in player_name_map:
                        new_id = player_name_map[name]
                        id_mapping_2023_to_2024[old_id] = new_id
                
                # Update 2023 historical data with consistent IDs
                hist_2023['element'] = hist_2023['element'].map(id_mapping_2023_to_2024).fillna(hist_2023['element'])
                print(f"Mapped {len(id_mapping_2023_to_2024)} players from 2023 to 2024 IDs")
                
            except FileNotFoundError:
                print("2023 elements file not found - keeping original 2023 IDs")
        
        seasons_data.append((hist_2023, fixtures_2023))
        print(f"Loaded 2023 data: {hist_2023.shape[0]} historical records")
    except FileNotFoundError:
        print("2023 data not found - using 2024 data only")
    
    return seasons_data, elements_df

# Load data from multiple seasons with player matching
seasons_data, elements_df = load_multi_season_data()

# If only one season available, use single season approach
if len(seasons_data) == 1:
    historical_df = seasons_data[0][0]
    future_fixtures_df = seasons_data[0][1]
else:
    # Combine multiple seasons
    historical_dfs = [data[0] for data in seasons_data]
    fixture_dfs = [data[1] for data in seasons_data]
    
    historical_df = pd.concat(historical_dfs, ignore_index=True)
    future_fixtures_df = pd.concat(fixture_dfs, ignore_index=True)

print(f"\nCombined historical data shape: {historical_df.shape}")
print(f"Combined fixtures data shape: {future_fixtures_df.shape}")

# Show player mapping stats if available
if elements_df is not None:
    print(f"Player elements loaded: {len(elements_df)} players")
    print(f"Sample players: {elements_df[['full_name', 'id', 'team']].head()}")

print(f"\nTarget variable (total_points) stats:")
print(historical_df['total_points'].describe())

#%%
def create_player_mapping(historical_df):
    """
    Create player mapping based on names (if available) or use element IDs
    Note: This is a simplified approach - you may need actual player name data
    """
    # For now, we'll use element IDs but add season info
    # In practice, you'd want to match on firstname + surname
    player_mapping = {}
    
    for _, row in historical_df.iterrows():
        season = row.get('season', 2024)
        element_id = row['element']
        
        # Create a unique key (you'd replace this with firstname_surname)
        player_key = f"player_{element_id}_season_{season}"
        
        if player_key not in player_mapping:
            player_mapping[player_key] = {
                'element_id': element_id,
                'season': season,
                # Add firstname, surname here when available
            }
    
    return player_mapping

def create_future_target(df, target_gameweeks=1):
    """
    Create target variable: next gameweek(s) total_points
    
    Args:
        df: Historical dataframe
        target_gameweeks: Number of future gameweeks to average (1 for t+1, 4 for t+1 to t+4)
    
    Returns:
        DataFrame with future targets
    """
    df_sorted = df.sort_values(['season', 'element', 'round'])
    
    targets = []
    for season in df_sorted['season'].unique():
        season_data = df_sorted[df_sorted['season'] == season]
        
        for element_id in season_data['element'].unique():
            element_data = season_data[season_data['element'] == element_id].copy()
            element_data = element_data.sort_values('round')
            
            # Create future targets within the same season
            for i in range(len(element_data)):
                future_points = []
                current_round = element_data.iloc[i]['round']
                
                # Get points from next target_gameweeks within same season
                for j in range(1, target_gameweeks + 1):
                    future_round = current_round + j
                    future_match = element_data[element_data['round'] == future_round]
                    
                    if len(future_match) > 0:
                        future_points.append(future_match.iloc[0]['total_points'])
                
                # Calculate target (average if multiple gameweeks)
                if future_points:
                    if target_gameweeks == 1:
                        target = future_points[0]
                    else:
                        target = np.mean(future_points)
                    
                    targets.append({
                        'element': element_id,
                        'round': current_round,
                        'season': season,
                        'future_points': target,
                        'future_gameweeks_available': len(future_points)
                    })
    
    return pd.DataFrame(targets)

#%%
def get_fixture_difficulty(element_id, round_num, season, fixtures_df, historical_df, target_gameweeks=1):
    """
    Get fixture difficulty for upcoming matches (season-aware)
    """
    # Filter fixtures for the specific season
    season_fixtures = fixtures_df[fixtures_df['season'] == season]
    
    # Get fixtures for the target gameweeks
    future_rounds = list(range(round_num + 1, round_num + target_gameweeks + 1))
    
    difficulties = []
    home_matches = 0
    
    for future_round in future_rounds:
        # Get fixtures for this round
        if 'event' in season_fixtures.columns:
            round_fixtures = season_fixtures[season_fixtures['event'] == future_round]
        else:
            # Handle different fixture file formats
            continue
            
        if len(round_fixtures) > 0:
            if 'difficulty' in round_fixtures.columns:
                avg_difficulty = round_fixtures['difficulty'].mean()
            else:
                # Calculate from team difficulties if available
                h_diff = round_fixtures.get('team_h_difficulty', 3).mean()
                a_diff = round_fixtures.get('team_a_difficulty', 3).mean()
                avg_difficulty = (h_diff + a_diff) / 2
                
            difficulties.append(avg_difficulty)
            
            # Count home matches
            if 'is_home' in round_fixtures.columns:
                home_count = round_fixtures['is_home'].sum()
                home_matches += home_count / len(round_fixtures)
    
    avg_difficulty = np.mean(difficulties) if difficulties else 3
    home_ratio = home_matches / target_gameweeks if target_gameweeks > 0 else 0
    
    return avg_difficulty, home_ratio

#%%
def create_enhanced_features(historical_df, future_fixtures_df, target_gameweeks=1):
    """
    Create enhanced feature set including future fixture difficulty
    """
    # Create future targets
    print(f"Creating targets for next {target_gameweeks} gameweek(s)...")
    future_targets = create_future_target(historical_df, target_gameweeks)
    
    # Merge with historical data
    enhanced_df = historical_df.merge(
        future_targets, 
        on=['element', 'round', 'season'], 
        how='inner'
    )
    
    print(f"Merged dataset shape: {enhanced_df.shape}")
    print(f"Samples with future targets: {len(enhanced_df)}")
    
    # Add datetime features
    enhanced_df['kickoff_time'] = pd.to_datetime(enhanced_df['kickoff_time'])
    enhanced_df['hour'] = enhanced_df['kickoff_time'].dt.hour
    enhanced_df['day_of_week'] = enhanced_df['kickoff_time'].dt.dayofweek
    enhanced_df['month'] = enhanced_df['kickoff_time'].dt.month
    
    # Convert boolean to int
    enhanced_df['was_home'] = enhanced_df['was_home'].astype(int)
    enhanced_df['modified'] = enhanced_df['modified'].astype(int)
    
    # Create rolling averages (last 3 games performance within season)
    enhanced_df = enhanced_df.sort_values(['season', 'element', 'round'])
    
    rolling_features = ['total_points', 'minutes', 'goals_scored', 'assists', 
                       'expected_goals', 'expected_assists', 'ict_index', 'bps']
    
    for feature in rolling_features:
        if feature in enhanced_df.columns:
            enhanced_df[f'{feature}_last3'] = (
                enhanced_df.groupby(['season', 'element'])[feature]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=[0,1], drop=True)
            )
    
    # Add fixture difficulty for upcoming matches
    print("Adding fixture difficulty features...")
    enhanced_df['next_fixture_difficulty'] = 0
    enhanced_df['next_home_ratio'] = 0
    
    for idx, row in enhanced_df.iterrows():
        difficulty, home_ratio = get_fixture_difficulty(
            row['element'], row['round'], row['season'], future_fixtures_df, historical_df, target_gameweeks
        )
        enhanced_df.at[idx, 'next_fixture_difficulty'] = difficulty
        enhanced_df.at[idx, 'next_home_ratio'] = home_ratio
    
    # Create form indicators
    enhanced_df['recent_form'] = enhanced_df['total_points_last3']
    enhanced_df['goals_form'] = enhanced_df['goals_scored_last3']
    enhanced_df['assists_form'] = enhanced_df['assists_last3']
    
    # Team strength indicators (season-aware)
    enhanced_df['team_goals_scored'] = enhanced_df.groupby(['season', 'round', 'opponent_team'])['goals_scored'].transform('sum')
    enhanced_df['team_goals_conceded'] = enhanced_df.groupby(['season', 'round', 'opponent_team'])['goals_conceded'].transform('sum')
    
    return enhanced_df

#%%
# Create enhanced dataset
TARGET_GAMEWEEKS = 4  # Change to 4 for t+1 to t+4 average
enhanced_data = create_enhanced_features(historical_df, future_fixtures_df, TARGET_GAMEWEEKS)

# Select features for training
feature_columns = [
    # Basic info
    'element', 'opponent_team', 'was_home', 'round', 'season',
    
    # Current game stats
    'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
    'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'starts',
    
    # Expected stats
    'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
    
    # Market data
    'value', 'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
    
    # Time features
    'hour', 'day_of_week', 'month',
    
    # Rolling form features
    'total_points_last3', 'minutes_last3', 'goals_scored_last3', 'assists_last3',
    'expected_goals_last3', 'expected_assists_last3', 'ict_index_last3', 'bps_last3',
    
    # Future fixture features
    'next_fixture_difficulty', 'next_home_ratio',
    
    # Form indicators
    'recent_form', 'goals_form', 'assists_form',
    
    # Team strength
    'team_goals_scored', 'team_goals_conceded'
]

# Filter to available columns
available_features = [col for col in feature_columns if col in enhanced_data.columns]
print(f"\nUsing {len(available_features)} features")

# Prepare training data
X = enhanced_data[available_features] #.fillna(0)
y = enhanced_data['future_points']

# Remove rows with missing targets
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

print(f"Final dataset for training: {X.shape[0]} samples")
print(f"Target variable stats (next {TARGET_GAMEWEEKS} gameweek(s)):")
print(y.describe())

#%%
# Split data into train/validation/test sets
if 'season' in enhanced_data.columns and len(enhanced_data['season'].unique()) > 1:
    print("Using season-based split")
    # Train: 2023, Validation: Early 2024, Test: Late 2024
    train_mask = enhanced_data['season'] == 2023
    
    # Split 2024 data into validation and test
    data_2024 = enhanced_data[enhanced_data['season'] == 2024]
    split_round_2024 = data_2024['round'].quantile(0.5)  # Split 2024 in half
    
    val_mask = (enhanced_data['season'] == 2024) & (enhanced_data['round'] <= split_round_2024)
    test_mask = (enhanced_data['season'] == 2024) & (enhanced_data['round'] > split_round_2024)
    
    print(f"Train: 2023 season")
    print(f"Validation: 2024 rounds 1-{int(split_round_2024)}")
    print(f"Test: 2024 rounds {int(split_round_2024)+1}+")
    
else:
    # Fallback: temporal split within single season (60/20/20)
    print("Using temporal split within season")
    split_1 = enhanced_data['round'].quantile(0.6)
    split_2 = enhanced_data['round'].quantile(0.8)
    
    train_mask = enhanced_data['round'] <= split_1
    val_mask = (enhanced_data['round'] > split_1) & (enhanced_data['round'] <= split_2)
    test_mask = enhanced_data['round'] > split_2

train_mask = train_mask[valid_mask]
val_mask = val_mask[valid_mask]
test_mask = test_mask[valid_mask]

X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

print(f"\nData splits:")
print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples") 
print(f"Testing: {len(X_test)} samples")

#%%
#%%
# Hyperparameter tuning with validation set
def tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_iter=50):
    """
    Tune XGBoost hyperparameters using validation set
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_iter: Number of parameter combinations to try
    
    Returns:
        Best model and parameters
    """
    # Define parameter search space
    param_distributions = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    best_rmse = float('inf')
    best_params = None
    best_model = None
    
    print(f"Starting hyperparameter tuning with {n_iter} iterations...")
    
    # Random search
    np.random.seed(42)
    for i in range(n_iter):
        # Sample random parameters
        params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        for param, values in param_distributions.items():
            params[param] = np.random.choice(values)
        
        # Train model with current parameters
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # Track best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params.copy()
            best_model = model
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{n_iter}, Best validation RMSE: {best_rmse:.4f}")
    
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        if param not in ['objective', 'random_state', 'n_jobs']:
            print(f"  {param}: {value}")
    
    print(f"Best validation RMSE: {best_rmse:.4f}")
    
    return best_model, best_params

# Train XGBoost model
print(f"\nTraining XGBoost model to predict next {TARGET_GAMEWEEKS} gameweek(s)...")

# Option 1: Use hyperparameter tuning (recommended but slower)
USE_HYPERPARAMETER_TUNING = True  # Set to False for faster training

if USE_HYPERPARAMETER_TUNING:
    print("Using hyperparameter tuning with validation set...")
    model, best_params = tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_iter=30)
else:
    # Option 2: Use default parameters (faster)
    print("Using default parameters...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

# Make predictions on all sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate metrics
def evaluate_model(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{dataset_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return rmse, mae, r2

train_rmse, train_mae, train_r2 = evaluate_model(y_train, y_train_pred, "Training")
val_rmse, val_mae, val_r2 = evaluate_model(y_val, y_val_pred, "Validation")
test_rmse, test_mae, test_r2 = evaluate_model(y_test, y_test_pred, "Test")

#%%
# Feature importance
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features for Predicting Next {TARGET_GAMEWEEKS} Gameweek(s):")
print(importance_df.head(15))

#%%
# Prediction function for future gameweeks
def predict_future_points(model, player_data, upcoming_fixtures, element_id, target_round, target_gameweeks=1, elements_df=None):
    """
    Predict points for a player in future gameweeks
    
    Args:
        model: Trained XGBoost model
        player_data: Historical data for the player
        upcoming_fixtures: Future fixtures data
        element_id: Player ID
        target_round: Round to predict for
        target_gameweeks: Number of gameweeks to predict
        elements_df: Player elements data for additional features
    
    Returns:
        Predicted points
    """
    # Get latest stats for player
    player_recent = player_data[player_data['element'] == element_id].tail(5)
    
    if len(player_recent) == 0:
        return 0  # No historical data
    
    # Calculate recent form (only numeric columns)
    numeric_columns = player_recent.select_dtypes(include=[np.number]).columns
    recent_stats = player_recent[numeric_columns].mean()
    
    # Get player info from elements if available
    player_info = {}
    if elements_df is not None:
        player_row = elements_df[elements_df['id'] == element_id]
        if len(player_row) > 0:
            player_info = {
                'position': player_row.iloc[0].get('element_type', 2),
                'team': player_row.iloc[0].get('team', 10),
                'form': player_row.iloc[0].get('form', 0),
                'now_cost': player_row.iloc[0].get('now_cost', 50),
                'selected_by_percent': player_row.iloc[0].get('selected_by_percent', 5.0)
            }
    
    # Get fixture difficulty for upcoming matches
    current_season = player_recent['season'].iloc[-1] if 'season' in player_recent.columns else 2024
    upcoming_difficulty = upcoming_fixtures[
        (upcoming_fixtures['event'] >= target_round) & 
        (upcoming_fixtures['event'] < target_round + target_gameweeks) &
        (upcoming_fixtures['season'] == current_season)
    ]['difficulty'].mean()
    
    home_fixtures = upcoming_fixtures[
        (upcoming_fixtures['event'] >= target_round) & 
        (upcoming_fixtures['event'] < target_round + target_gameweeks) &
        (upcoming_fixtures['season'] == current_season)
    ]
    
    home_ratio = 0
    if len(home_fixtures) > 0 and 'is_home' in home_fixtures.columns:
        home_ratio = home_fixtures['is_home'].sum() / len(home_fixtures)
    
    # Create prediction input with enhanced features
    prediction_input = pd.DataFrame([{
        'element': element_id,
        'opponent_team': recent_stats.get('opponent_team', 10),
        'was_home': 1 if home_ratio > 0.5 else 0,
        'round': target_round,
        'season': current_season,
        'minutes': recent_stats.get('minutes', 70),
        'goals_scored': recent_stats.get('goals_scored', 0),
        'assists': recent_stats.get('assists', 0),
        'clean_sheets': recent_stats.get('clean_sheets', 0),
        'goals_conceded': recent_stats.get('goals_conceded', 1),
        'own_goals': recent_stats.get('own_goals', 0),
        'penalties_saved': recent_stats.get('penalties_saved', 0),
        'penalties_missed': recent_stats.get('penalties_missed', 0),
        'yellow_cards': recent_stats.get('yellow_cards', 0),
        'red_cards': recent_stats.get('red_cards', 0),
        'saves': recent_stats.get('saves', 0),
        'bonus': recent_stats.get('bonus', 0),
        'bps': recent_stats.get('bps', 20),
        'influence': recent_stats.get('influence', 50),
        'creativity': recent_stats.get('creativity', 30),
        'threat': recent_stats.get('threat', 40),
        'ict_index': recent_stats.get('ict_index', 120),
        'starts': recent_stats.get('starts', 1),
        'expected_goals': recent_stats.get('expected_goals', 0.3),
        'expected_assists': recent_stats.get('expected_assists', 0.2),
        'expected_goal_involvements': recent_stats.get('expected_goal_involvements', 0.5),
        'expected_goals_conceded': recent_stats.get('expected_goals_conceded', 1.2),
        'value': recent_stats.get('value', 60),
        'transfers_balance': recent_stats.get('transfers_balance', 0),
        'selected': recent_stats.get('selected', 1000),
        'transfers_in': recent_stats.get('transfers_in', 100),
        'transfers_out': recent_stats.get('transfers_out', 100),
        'hour': 15,
        'day_of_week': 5,
        'month': 10,
        'total_points_last3': recent_stats.get('total_points', 2),
        'minutes_last3': recent_stats.get('minutes', 70),
        'goals_scored_last3': recent_stats.get('goals_scored', 0),
        'assists_last3': recent_stats.get('assists', 0),
        'expected_goals_last3': recent_stats.get('expected_goals', 0.3),
        'expected_assists_last3': recent_stats.get('expected_assists', 0.2),
        'ict_index_last3': recent_stats.get('ict_index', 120),
        'bps_last3': recent_stats.get('bps', 20),
        'next_fixture_difficulty': upcoming_difficulty if not pd.isna(upcoming_difficulty) else 3,
        'next_home_ratio': home_ratio,
        'recent_form': recent_stats.get('total_points', 2),
        'goals_form': recent_stats.get('goals_scored', 0),
        'assists_form': recent_stats.get('assists', 0),
        'team_goals_scored': recent_stats.get('team_goals_scored', 1),
        'team_goals_conceded': recent_stats.get('team_goals_conceded', 1)
    }])
    
    # Ensure all features are present
    for feature in available_features:
        if feature not in prediction_input.columns:
            prediction_input[feature] = 0
    
    # Reorder columns to match training
    prediction_input = prediction_input[available_features].fillna(0)
    
    # Make prediction
    predicted_points = model.predict(prediction_input)[0]
    return predicted_points, player_info

#%%
# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS FOR NEXT GAMEWEEK")
print("="*60)

# Get some example players
example_elements = enhanced_data['element'].unique()[:5]

for element_id in example_elements:
    predicted, player_info = predict_future_points(
        model, historical_df, future_fixtures_df, 
        element_id, target_round=20, target_gameweeks=TARGET_GAMEWEEKS,
        elements_df=elements_df
    )
    
    # Get player name if available
    player_name = "Unknown"
    if elements_df is not None:
        player_row = elements_df[elements_df['id'] == element_id]
        if len(player_row) > 0:
            player_name = f"{player_row.iloc[0]['first_name']} {player_row.iloc[0]['second_name']}"
    
    print(f"Element {element_id} ({player_name}): Predicted points = {predicted:.2f}")

#%%
# Model summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Model: XGBoost Regressor (Future Points Prediction)")
print(f"Target: Next {TARGET_GAMEWEEKS} gameweek(s) average points")
print(f"Features: {len(available_features)}")
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE: {val_mae:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Save the model
model.save_model(f'fantasy_football_future_xgb_gw{TARGET_GAMEWEEKS}.json')
print(f"\nModel saved as 'fantasy_football_future_xgb_gw{TARGET_GAMEWEEKS}.json'")

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print("Most important features for predicting future performance:")
top_features = importance_df.head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:25} (importance: {row['importance']:.4f})")

print(f"\nNote: Change TARGET_GAMEWEEKS to 4 to predict average of next 4 gameweeks")
print(f"Current setting: Predicting next {TARGET_GAMEWEEKS} gameweek(s)")