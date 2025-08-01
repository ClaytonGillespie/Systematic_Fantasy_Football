#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

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
        print(f"Loaded 2024 player elements data: {elements_df.shape[0]} players")
        
        # Create player name to ID mapping
        elements_df['full_name'] = elements_df['first_name'].str.strip() + '_' + elements_df['second_name'].str.strip()
        player_name_map = dict(zip(elements_df['full_name'], elements_df['id']))
        
        print(f"Sample player names: {list(elements_df['full_name'].head())}")
        
    except FileNotFoundError:
        print("2024 player elements file not found - using element IDs directly")
        elements_df = None
        player_name_map = {}
    
    # Load 2024 data
    try:
        hist_2024 = pd.read_csv('./data_2024/previous_fixtures_38.csv')
        fixtures_2024 = pd.read_csv('./data_2024/future_fixtures_1.csv')
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
        
        # Standardize fixtures_2023 to match future_fixtures format (if needed)
        if 'difficulty' not in fixtures_2023.columns and 'team_h_difficulty' in fixtures_2023.columns:
            fixtures_2023['difficulty'] = (fixtures_2023['team_h_difficulty'] + fixtures_2023['team_a_difficulty']) / 2
        
        # Load 2023 elements for player name matching
        if elements_df is not None:
            try:
                elements_2023 = pd.read_csv('./data_2023/elements_38.csv')
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
                print("2023 elements file (elements_38.csv) not found - keeping original 2023 IDs")
        
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
def create_future_target(df, target_gameweeks=1):
    """
    Create target variable: next gameweek(s) total_points
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
    'opponent_team', 'was_home', 'round', 'season',
    
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
X = enhanced_data[available_features].fillna(0)
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
# Hyperparameter tuning with validation set
def tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_iter=50):
    """
    Tune XGBoost hyperparameters using validation set
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
    
    # Calculate Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    spearman_r2 = spearman_corr ** 2
    
    print(f"\n{dataset_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Spearman ρ: {spearman_corr:.4f}")
    print(f"Spearman R²: {spearman_r2:.4f}")
    
    return rmse, mae, r2, spearman_corr, spearman_r2

train_rmse, train_mae, train_r2, train_spearman, train_spearman_r2 = evaluate_model(y_train, y_train_pred, "Training")
val_rmse, val_mae, val_r2, val_spearman, val_spearman_r2 = evaluate_model(y_val, y_val_pred, "Validation")
test_rmse, test_mae, test_r2, test_spearman, test_spearman_r2 = evaluate_model(y_test, y_test_pred, "Test")

#%%
# Feature importance
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features for Predicting Next {TARGET_GAMEWEEKS} Gameweek(s):")
print(importance_df.head(15))

#%%
# Create comprehensive predictions DataFrame for visualization
def create_predictions_dataframe(model, X_data, y_data, enhanced_data, mask, elements_df, dataset_name):
    """
    Create a comprehensive DataFrame with predictions, player info, and features
    """
    # Get predictions
    predictions = model.predict(X_data)
    
    # Get the corresponding rows from enhanced_data using integer indexing
    mask_indices = enhanced_data.index[mask].tolist()
    data_subset = enhanced_data.iloc[mask_indices].copy()
    
    # Add predictions and actual values
    data_subset = data_subset.reset_index(drop=True)
    data_subset['predicted_points'] = predictions
    data_subset['actual_points'] = y_data.reset_index(drop=True)
    data_subset['prediction_error'] = data_subset['actual_points'] - data_subset['predicted_points']
    data_subset['absolute_error'] = abs(data_subset['prediction_error'])
    
    # Add player names if elements data available
    if elements_df is not None:
        # Create player info mapping
        player_info = elements_df[['id', 'first_name', 'second_name', 'web_name', 'element_type', 'team', 'now_cost']].copy()
        player_info = player_info.rename(columns={'id': 'element'})
        
        # Merge with predictions
        data_subset = data_subset.merge(player_info, on='element', how='left')
        data_subset['player_name'] = data_subset['first_name'].fillna('') + ' ' + data_subset['second_name'].fillna('')
        data_subset['player_name'] = data_subset['player_name'].str.strip()
        
        # Fill missing names with web_name or element ID
        data_subset['player_name'] = data_subset['player_name'].replace('', None)
        data_subset['player_name'] = data_subset['player_name'].fillna(data_subset['web_name'])
        data_subset['player_name'] = data_subset['player_name'].fillna('Player_' + data_subset['element'].astype(str))
    else:
        data_subset['player_name'] = 'Player_' + data_subset['element'].astype(str)
        data_subset['element_type'] = None
        data_subset['team'] = None
        data_subset['now_cost'] = None
    
    # Add fixture difficulty and upcoming opponent info
    data_subset['fixture_info'] = (
        'R' + data_subset['round'].astype(str) + 
        ' vs Team' + data_subset['opponent_team'].astype(str) + 
        data_subset['was_home'].apply(lambda x: ' (H)' if x else ' (A)')
    )
    
    # Select key columns for visualization
    viz_columns = [
        'player_name', 'element', 'element_type', 'team', 'now_cost',
        'season', 'round', 'fixture_info', 'opponent_team', 'was_home',
        'actual_points', 'predicted_points', 'prediction_error', 'absolute_error',
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'bonus', 'bps', 'ict_index', 'expected_goals', 'expected_assists',
        'total_points_last3', 'recent_form', 'next_fixture_difficulty', 'next_home_ratio',
        'value', 'transfers_balance', 'selected'
    ]
    
    # Only keep columns that exist
    available_viz_columns = [col for col in viz_columns if col in data_subset.columns]
    
    result_df = data_subset[available_viz_columns].copy()
    
    # Sort by absolute error (worst predictions first for analysis)
    result_df = result_df.sort_values('absolute_error', ascending=False)
    
    print(f"\n{dataset_name} predictions DataFrame created:")
    print(f"Shape: {result_df.shape}")
    
    return result_df

# Create prediction DataFrames for each split
print("Creating comprehensive prediction DataFrames...")

train_predictions_df = create_predictions_dataframe(
    model, X_train, y_train, enhanced_data, train_mask, elements_df, "Training"
)

val_predictions_df = create_predictions_dataframe(
    model, X_val, y_val, enhanced_data, val_mask, elements_df, "Validation"
)

test_predictions_df = create_predictions_dataframe(
    model, X_test, y_test, enhanced_data, test_mask, elements_df, "Test"
)

# Combine all predictions for comprehensive analysis
all_predictions_df = pd.concat([
    train_predictions_df.assign(dataset='train'),
    val_predictions_df.assign(dataset='validation'), 
    test_predictions_df.assign(dataset='test')
], ignore_index=True)

print(f"\nCombined predictions DataFrame: {all_predictions_df.shape}")

#%%
# Function to analyze R² performance across different target gameweeks
def analyze_r2_by_target_gameweeks(historical_df, future_fixtures_df, elements_df, max_gameweeks=6):
    """
    Analyze how R² performance changes with different TARGET_GAMEWEEKS
    """
    results = []
    
    print(f"Analyzing R² performance for 1 to {max_gameweeks} target gameweeks...")
    print("This may take several minutes...")
    
    for target_gw in range(1, max_gameweeks + 1):
        print(f"\nTesting TARGET_GAMEWEEKS = {target_gw}")
        
        try:
            # Create enhanced features for this target gameweek setting
            enhanced_data_temp = create_enhanced_features(historical_df, future_fixtures_df, target_gw)
            
            # Prepare data
            X_temp = enhanced_data_temp[available_features].fillna(0)
            y_temp = enhanced_data_temp['future_points']
            
            # Remove rows with missing targets
            valid_mask_temp = ~y_temp.isna()
            X_temp = X_temp[valid_mask_temp]
            y_temp = y_temp[valid_mask_temp]
            
            if len(X_temp) < 100:  # Skip if not enough data
                print(f"  Insufficient data ({len(X_temp)} samples) - skipping")
                continue
            
            # Split data (same logic as main model)
            if 'season' in enhanced_data_temp.columns and len(enhanced_data_temp['season'].unique()) > 1:
                train_mask_temp = enhanced_data_temp['season'] == 2023
                data_2024_temp = enhanced_data_temp[enhanced_data_temp['season'] == 2024]
                
                if len(data_2024_temp) > 0:
                    split_round_2024_temp = data_2024_temp['round'].quantile(0.5)
                    val_mask_temp = (enhanced_data_temp['season'] == 2024) & (enhanced_data_temp['round'] <= split_round_2024_temp)
                    test_mask_temp = (enhanced_data_temp['season'] == 2024) & (enhanced_data_temp['round'] > split_round_2024_temp)
                else:
                    # Fallback if no 2024 data
                    split_1 = enhanced_data_temp['round'].quantile(0.6)
                    split_2 = enhanced_data_temp['round'].quantile(0.8)
                    train_mask_temp = enhanced_data_temp['round'] <= split_1
                    val_mask_temp = (enhanced_data_temp['round'] > split_1) & (enhanced_data_temp['round'] <= split_2)
                    test_mask_temp = enhanced_data_temp['round'] > split_2
            else:
                # Single season split
                split_1 = enhanced_data_temp['round'].quantile(0.6)
                split_2 = enhanced_data_temp['round'].quantile(0.8)
                train_mask_temp = enhanced_data_temp['round'] <= split_1
                val_mask_temp = (enhanced_data_temp['round'] > split_1) & (enhanced_data_temp['round'] <= split_2)
                test_mask_temp = enhanced_data_temp['round'] > split_2
            
            # Apply valid mask
            train_mask_temp = train_mask_temp[valid_mask_temp]
            val_mask_temp = val_mask_temp[valid_mask_temp]
            test_mask_temp = test_mask_temp[valid_mask_temp]
            
            X_train_temp = X_temp[train_mask_temp]
            X_val_temp = X_temp[val_mask_temp]
            X_test_temp = X_temp[test_mask_temp]
            y_train_temp = y_temp[train_mask_temp]
            y_val_temp = y_temp[val_mask_temp]
            y_test_temp = y_temp[test_mask_temp]
            
            if len(X_train_temp) < 50 or len(X_test_temp) < 10:
                print(f"  Insufficient split data - skipping")
                continue
            
            # Train a simple model (no hyperparameter tuning for speed)
            model_temp = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model_temp.fit(X_train_temp, y_train_temp)
            
            # Evaluate on all sets
            y_train_pred_temp = model_temp.predict(X_train_temp)
            y_val_pred_temp = model_temp.predict(X_val_temp)
            y_test_pred_temp = model_temp.predict(X_test_temp)
            
            # Calculate R² and Spearman scores
            train_r2_temp = r2_score(y_train_temp, y_train_pred_temp)
            val_r2_temp = r2_score(y_val_temp, y_val_pred_temp) if len(y_val_temp) > 0 else 0
            test_r2_temp = r2_score(y_test_temp, y_test_pred_temp)
            
            # Calculate Spearman correlations
            train_spearman_temp, _ = spearmanr(y_train_temp, y_train_pred_temp)
            val_spearman_temp, _ = spearmanr(y_val_temp, y_val_pred_temp) if len(y_val_temp) > 0 else (0, 1)
            test_spearman_temp, _ = spearmanr(y_test_temp, y_test_pred_temp)
            
            train_spearman_r2_temp = train_spearman_temp ** 2
            val_spearman_r2_temp = val_spearman_temp ** 2
            test_spearman_r2_temp = test_spearman_temp ** 2
            
            # Calculate RMSE for additional insight
            train_rmse_temp = np.sqrt(mean_squared_error(y_train_temp, y_train_pred_temp))
            val_rmse_temp = np.sqrt(mean_squared_error(y_val_temp, y_val_pred_temp)) if len(y_val_temp) > 0 else 0
            test_rmse_temp = np.sqrt(mean_squared_error(y_test_temp, y_test_pred_temp))
            
            results.append({
                'target_gameweeks': target_gw,
                'train_samples': len(X_train_temp),
                'val_samples': len(X_val_temp), 
                'test_samples': len(X_test_temp),
                'train_r2': train_r2_temp,
                'val_r2': val_r2_temp,
                'test_r2': test_r2_temp,
                'train_spearman': train_spearman_temp,
                'val_spearman': val_spearman_temp,
                'test_spearman': test_spearman_temp,
                'train_spearman_r2': train_spearman_r2_temp,
                'val_spearman_r2': val_spearman_r2_temp,
                'test_spearman_r2': test_spearman_r2_temp,
                'train_rmse': train_rmse_temp,
                'val_rmse': val_rmse_temp,
                'test_rmse': test_rmse_temp,
                'target_mean': y_temp.mean(),
                'target_std': y_temp.std()
            })
            
            print(f"  Train R²: {train_r2_temp:.4f}, Val R²: {val_r2_temp:.4f}, Test R²: {test_r2_temp:.4f}")
            print(f"  Train Spearman R²: {train_spearman_r2_temp:.4f}, Val Spearman R²: {val_spearman_r2_temp:.4f}, Test Spearman R²: {test_spearman_r2_temp:.4f}")
            print(f"  Train RMSE: {train_rmse_temp:.4f}, Val RMSE: {val_rmse_temp:.4f}, Test RMSE: {test_rmse_temp:.4f}")
            
        except Exception as e:
            print(f"  Error with target_gw={target_gw}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Run the R² analysis
r2_analysis_df = analyze_r2_by_target_gameweeks(historical_df, future_fixtures_df, elements_df, max_gameweeks=6)

# Display and save R² analysis results
if len(r2_analysis_df) > 0:
    print("\n" + "="*80)
    print("R² ANALYSIS BY TARGET GAMEWEEKS")
    print("="*80)
    
    # Display formatted results
    print("\nR² Performance Summary:")
    print(f"{'GW':>3} {'Train R²':>9} {'Val R²':>7} {'Test R²':>8} {'Test Spear R²':>12} {'Train RMSE':>11} {'Test RMSE':>10} {'Samples':>8}")
    print("-" * 90)
    
    for _, row in r2_analysis_df.iterrows():
        print(f"{row['target_gameweeks']:3.0f} "
              f"{row['train_r2']:9.4f} "
              f"{row['val_r2']:7.4f} "
              f"{row['test_r2']:8.4f} "
              f"{row['test_spearman_r2']:12.4f} "
              f"{row['train_rmse']:11.4f} "
              f"{row['test_rmse']:10.4f} "
              f"{row['test_samples']:8.0f}")
    
    # Find optimal target gameweeks
    best_test_r2_idx = r2_analysis_df['test_r2'].idxmax()
    best_test_spearman_r2_idx = r2_analysis_df['test_spearman_r2'].idxmax()
    
    print(f"\nOptimal Settings:")
    print(f"Best Test R²: {r2_analysis_df.loc[best_test_r2_idx, 'test_r2']:.4f} at {r2_analysis_df.loc[best_test_r2_idx, 'target_gameweeks']:.0f} gameweeks")
    print(f"Best Test Spearman R²: {r2_analysis_df.loc[best_test_spearman_r2_idx, 'test_spearman_r2']:.4f} at {r2_analysis_df.loc[best_test_spearman_r2_idx, 'target_gameweeks']:.0f} gameweeks")
    
    # Save results
    r2_analysis_df.to_csv('r2_analysis_by_target_gameweeks.csv', index=False)
    print(f"\nR² analysis saved to 'r2_analysis_by_target_gameweeks.csv'")
    
    # Save predictions to CSV for external visualization
    test_predictions_df.to_csv(f'test_predictions_gw{TARGET_GAMEWEEKS}.csv', index=False)
    all_predictions_df.to_csv(f'all_predictions_gw{TARGET_GAMEWEEKS}.csv', index=False)
    
    print(f"\nPrediction DataFrames saved:")
    print(f"- test_predictions_gw{TARGET_GAMEWEEKS}.csv")
    print(f"- all_predictions_gw{TARGET_GAMEWEEKS}.csv")
    print(f"- r2_analysis_by_target_gameweeks.csv")
    
    print(f"\nDataFrames available for visualization:")
    print(f"- test_predictions_df: {test_predictions_df.shape}")
    print(f"- val_predictions_df: {val_predictions_df.shape}")
    print(f"- train_predictions_df: {train_predictions_df.shape}")
    print(f"- all_predictions_df: {all_predictions_df.shape}")
    print(f"- r2_analysis_df: {r2_analysis_df.shape}")
    
else:
    print("No R² analysis results generated - check data availability")

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

print(f"\nNote: Change TARGET_GAMEWEEKS to 4 to predict average of next 4 gameweeks")
print(f"Current setting: Predicting next {TARGET_GAMEWEEKS} gameweek(s)")