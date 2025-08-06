# Fantasy Football Model Improvements

## Current Model Issues

1. **Missing function**: `analyze_target_distribution()` is called but not defined
2. **Poor predictions**: Top players like Salah getting ~1 point instead of ~36 points for 4 gameweeks
3. **Scale mismatch**: Model predictions don't align with expected FPL scoring ranges
4. **Target variable**: Predicting average over next 4 gameweeks may be too complex

## Suggested Improvements

### 1. Fix Missing Function
Add the missing `analyze_target_distribution` function:

```python
def analyze_target_distribution(y, target_name="target"):
    """Analyze target variable distribution for optimal objective selection"""
    import scipy.stats as stats
    
    y_clean = y.dropna()
    
    # Basic stats
    skewness = stats.skew(y_clean)
    kurtosis = stats.kurtosis(y_clean)
    zero_percent = (y_clean == 0).mean() * 100
    
    # Recommend objective based on distribution
    if zero_percent > 15:
        recommendation = 'tweedie'  # Good for zero-inflated data
    elif skewness > 1.5:
        recommendation = 'gamma'    # Good for right-skewed data
    else:
        recommendation = 'standard' # Standard MSE
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'zero_percent': zero_percent,
        'recommendation': recommendation
    }
```

### 2. Simplify Target Variable
Instead of predicting "average over next 4 gameweeks", predict "next single gameweek points":

```python
TARGET_GAMEWEEKS = 1  # Change from 4 to 1
```

Then for 4-gameweek predictions, multiply by 4 with some variance.

### 3. Better Feature Engineering

#### A. Use more recent form (last 5 games instead of 3):
```python
rolling_features = ['total_points', 'minutes', 'goals_scored', 'assists', 
                   'expected_goals', 'expected_assists', 'ict_index', 'bps']

for feature in rolling_features:
    if feature in enhanced_df.columns:
        enhanced_df[f'{feature}_last5'] = (
            enhanced_df.groupby(['season', 'element'])[feature]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=[0,1], drop=True)
        )
```

#### B. Add player-specific features:
```python
# Player consistency (coefficient of variation)
enhanced_df['points_consistency'] = (
    enhanced_df.groupby(['season', 'element'])['total_points']
    .transform(lambda x: x.std() / (x.mean() + 0.1))
)

# Player trend (improving vs declining)
enhanced_df['points_trend'] = (
    enhanced_df.groupby(['season', 'element'])['total_points']
    .transform(lambda x: x.iloc[-3:].mean() - x.iloc[:3].mean() if len(x) >= 6 else 0)
)
```

### 4. Improved Model Architecture

#### A. Ensemble approach:
```python
def create_ensemble_model(X_train, y_train, X_val, y_val):
    """Create ensemble of different model types"""
    
    models = {}
    
    # XGBoost with different objectives
    models['xgb_mse'] = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6, learning_rate=0.1, n_estimators=300
    )
    
    models['xgb_tweedie'] = xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=1.5,
        max_depth=6, learning_rate=0.1, n_estimators=300
    )
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"{name} validation RMSE: {val_rmse:.4f}")
    
    return models
```

#### B. Position-specific models:
```python
def train_position_specific_models(enhanced_df, available_features):
    """Train separate models for each position"""
    
    position_models = {}
    
    for position in [1, 2, 3, 4]:  # GKP, DEF, MID, FWD
        print(f"Training model for position {position}...")
        
        pos_data = enhanced_df[enhanced_df['element_type'] == position]
        
        if len(pos_data) < 100:
            continue
            
        X_pos = pos_data[available_features].fillna(0)
        y_pos = pos_data['future_points']
        
        # Position-specific hyperparameters
        if position == 1:  # Goalkeepers
            params = {'max_depth': 4, 'learning_rate': 0.05}
        elif position == 4:  # Forwards
            params = {'max_depth': 8, 'learning_rate': 0.1}
        else:
            params = {'max_depth': 6, 'learning_rate': 0.08}
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            **params
        )
        
        model.fit(X_pos, y_pos)
        position_models[position] = model
    
    return position_models
```

### 5. Better Validation Strategy

```python
# Use time-based cross-validation
from sklearn.model_selection import TimeSeriesSplit

def time_series_validation(enhanced_df, available_features, n_splits=5):
    """Time-based cross-validation for time series data"""
    
    # Sort by season and round
    enhanced_df = enhanced_df.sort_values(['season', 'round'])
    
    X = enhanced_df[available_features].fillna(0)
    y = enhanced_df['future_points']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6, learning_rate=0.1, n_estimators=300
        )
        
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        cv_scores.append(val_rmse)
        print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")
    
    print(f"Average CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return cv_scores
```

### 6. Model Interpretability

```python
def analyze_model_predictions(model, X_test, y_test, feature_names):
    """Analyze model predictions and feature importance"""
    
    predictions = model.predict(X_test)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Prediction analysis
    residuals = y_test - predictions
    
    print(f"\nPrediction Analysis:")
    print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.3f}")
    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")
    print(f"R² Score: {r2_score(y_test, predictions):.3f}")
    
    # Show best and worst predictions
    error_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'error': residuals,
        'abs_error': np.abs(residuals)
    }).sort_values('abs_error')
    
    print("\nBest Predictions (smallest errors):")
    print(error_df.head())
    
    print("\nWorst Predictions (largest errors):")
    print(error_df.tail())
    
    return importance_df
```

## Implementation Priority

1. **High Priority**: Fix missing function and retrain with TARGET_GAMEWEEKS=1
2. **Medium Priority**: Add position-specific models and better features
3. **Low Priority**: Implement ensemble and advanced validation

## Quick Fix for Current Issue

The immediate fix is to add the missing function and retrain:

```python
# Add this function to model.py before line 362
def analyze_target_distribution(y, target_name="target"):
    import scipy.stats as stats
    y_clean = y.dropna()
    return {
        'skewness': stats.skew(y_clean),
        'zero_percent': (y_clean == 0).mean() * 100,
        'recommendation': 'tweedie' if (y_clean == 0).mean() > 0.1 else 'standard'
    }
```

This should resolve the immediate training issues and produce a working model.