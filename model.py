#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
# Load the data
historical_df = pd.read_csv('./data_2024/previous_fixtures_38.csv')
future_fixtures_df = pd.read_csv('./data_2024/future_fixtures_1.csv')

print(f"Historical data shape: {historical_df.shape}")
print(f"Future fixtures shape: {future_fixtures_df.shape}")
print(f"Target variable (total_points) stats:")
print(historical_df['total_points'].describe())

#%%

def create_future_target(df, target_gameweeks=1):
    """
    Create target variable: next gameweek(s) total_points
    
    Args:
        df: Historical dataframe
        target_gameweeks: Number of future gameweeks to average (1 for t+1, 4 for t+1 to t+4)
    
    Returns:
        DataFrame with future targets
    """
    df_sorted = df.sort_values(['element', 'round'])
    
    targets = []
    for element_id in df_sorted['element'].unique():
        element_data = df_sorted[df_sorted['element'] == element_id].copy()
        element_data = element_data.sort_values('round')
        
        # Create future targets
        for i in range(len(element_data)):
            future_points = []
            current_round = element_data.iloc[i]['round']
            
            # Get points from next target_gameweeks
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
                    'future_points': target,
                    'future_gameweeks_available': len(future_points)
                })
    
    return pd.DataFrame(targets)

#%%
def get_fixture_difficulty(element_id, round_num, fixtures_df, historical_df, target_gameweeks=1):
    """
    Get fixture difficulty for upcoming matches
    """
    # Get team for this element from historical data
    element_team_matches = historical_df[historical_df['element'] == element_id]
    if len(element_team_matches) == 0:
        return 3, 0  # Default difficulty and home status
    
    # Find team ID (this is approximate - you might need to map this properly)
    # For now, we'll use opponent_team info to infer
    
    # Get fixtures for the target gameweeks
    future_rounds = list(range(round_num + 1, round_num + target_gameweeks + 1))
    
    difficulties = []
    home_matches = 0
    
    for future_round in future_rounds:
        # This is a simplified approach - you'd need proper team mapping
        # For demonstration, we'll use average difficulty
        round_fixtures = fixtures_df[fixtures_df['event'] == future_round]
        if len(round_fixtures) > 0:
            avg_difficulty = round_fixtures['difficulty'].mean()
            difficulties.append(avg_difficulty)
            # Count home matches (simplified)
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
        on=['element', 'round'], 
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
    
    # Create rolling averages (last 3 games performance)
    enhanced_df = enhanced_df.sort_values(['element', 'round'])
    
    rolling_features = ['total_points', 'minutes', 'goals_scored', 'assists', 
                       'expected_goals', 'expected_assists', 'ict_index', 'bps']
    
    for feature in rolling_features:
        if feature in enhanced_df.columns:
            enhanced_df[f'{feature}_last3'] = (
                enhanced_df.groupby('element')[feature]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    
    # Add fixture difficulty for upcoming matches
    print("Adding fixture difficulty features...")
    enhanced_df['next_fixture_difficulty'] = 0
    enhanced_df['next_home_ratio'] = 0
    
    for idx, row in enhanced_df.iterrows():
        difficulty, home_ratio = get_fixture_difficulty(
            row['element'], row['round'], future_fixtures_df, historical_df, target_gameweeks
        )
        enhanced_df.at[idx, 'next_fixture_difficulty'] = difficulty
        enhanced_df.at[idx, 'next_home_ratio'] = home_ratio
    
    # Create form indicators
    enhanced_df['recent_form'] = enhanced_df['total_points_last3']
    enhanced_df['goals_form'] = enhanced_df['goals_scored_last3']
    enhanced_df['assists_form'] = enhanced_df['assists_last3']
    
    # Team strength indicators (simplified)
    enhanced_df['team_goals_scored'] = enhanced_df.groupby(['round', 'opponent_team'])['goals_scored'].transform('sum')
    enhanced_df['team_goals_conceded'] = enhanced_df.groupby(['round', 'opponent_team'])['goals_conceded'].transform('sum')
    
    return enhanced_df

#%%
# Create enhanced dataset
TARGET_GAMEWEEKS = 4  # Change to 4 for t+1 to t+4 average
enhanced_data = create_enhanced_features(historical_df, future_fixtures_df, TARGET_GAMEWEEKS)

# Select features for training
feature_columns = [
    # Basic info
    'element', 'opponent_team', 'was_home', 'round',
    
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
# Split data temporally (use earlier rounds for training, later for testing)
split_round = enhanced_data['round'].quantile(0.8)
train_mask = enhanced_data['round'] <= split_round
test_mask = enhanced_data['round'] > split_round

train_mask = train_mask[valid_mask]
test_mask = test_mask[valid_mask]

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTemporal split:")
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

#%%
# Train XGBoost model
print(f"\nTraining XGBoost model to predict next {TARGET_GAMEWEEKS} gameweek(s)...")

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

# Make predictions
y_train_pred = model.predict(X_train)
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
def predict_future_points(model, player_data, upcoming_fixtures, element_id, target_round, target_gameweeks=1):
    """
    Predict points for a player in future gameweeks
    
    Args:
        model: Trained XGBoost model
        player_data: Historical data for the player
        upcoming_fixtures: Future fixtures data
        element_id: Player ID
        target_round: Round to predict for
        target_gameweeks: Number of gameweeks to predict
    
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
    
    # Get fixture difficulty for upcoming matches
    upcoming_difficulty = upcoming_fixtures[
        (upcoming_fixtures['event'] >= target_round) & 
        (upcoming_fixtures['event'] < target_round + target_gameweeks)
    ]['difficulty'].mean()
    
    home_ratio = upcoming_fixtures[
        (upcoming_fixtures['event'] >= target_round) & 
        (upcoming_fixtures['event'] < target_round + target_gameweeks) &
        (upcoming_fixtures['is_home'] == True)
    ].shape[0] / target_gameweeks
    
    # Create prediction input
    prediction_input = pd.DataFrame([{
        'element': element_id,
        'opponent_team': recent_stats.get('opponent_team', 10),
        'was_home': 1 if home_ratio > 0.5 else 0,
        'round': target_round,
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
    return predicted_points

#%%
# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS FOR NEXT GAMEWEEK")
print("="*60)

# Get some example players
example_elements = enhanced_data['element'].unique()[:5]

for element_id in example_elements:
    predicted = predict_future_points(
        model, historical_df, future_fixtures_df, 
        element_id, target_round=20, target_gameweeks=TARGET_GAMEWEEKS
    )
    print(f"Element {element_id}: Predicted points = {predicted:.2f}")

#%%
# Model summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Model: XGBoost Regressor (Future Points Prediction)")
print(f"Target: Next {TARGET_GAMEWEEKS} gameweek(s) average points")
print(f"Features: {len(available_features)}")
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
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