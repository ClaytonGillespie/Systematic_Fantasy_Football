#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#%%
# Analyze the distribution of fantasy points
def analyze_points_distribution(df, target_col='total_points'):
    """
    Analyze the distribution of fantasy football points
    """
    points = df[target_col].dropna()
    
    print("📊 FANTASY POINTS DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print(f"Count: {len(points):,}")
    print(f"Mean: {points.mean():.3f}")
    print(f"Median: {points.median():.3f}")
    print(f"Std: {points.std():.3f}")
    print(f"Skewness: {stats.skew(points):.3f}")
    print(f"Kurtosis: {stats.kurtosis(points):.3f}")
    
    # Percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p:2d}%: {np.percentile(points, p):5.1f}")
    
    # Zero/negative points
    zero_or_negative = (points <= 0).sum()
    print(f"\nZero or negative points: {zero_or_negative} ({zero_or_negative/len(points)*100:.1f}%)")
    
    # Test for log-normal distribution
    # Add small constant to handle zeros/negatives
    points_positive = points + 1  # Shift to make all positive
    
    # Fit log-normal distribution
    log_points = np.log(points_positive)
    mu, sigma = stats.norm.fit(log_points)
    
    print(f"\nLog-normal fit (after +1 shift):")
    print(f"  μ (log-space mean): {mu:.3f}")
    print(f"  σ (log-space std): {sigma:.3f}")
    
    # Kolmogorov-Smirnov test for log-normal
    ks_stat, ks_p = stats.kstest(log_points, 'norm', args=(mu, sigma))
    print(f"  KS test p-value: {ks_p:.6f}")
    print(f"  Log-normal fit: {'Good' if ks_p > 0.01 else 'Poor'}")
    
    return {
        'mean': points.mean(),
        'std': points.std(),
        'skewness': stats.skew(points),
        'log_mu': mu,
        'log_sigma': sigma,
        'ks_p_value': ks_p
    }

#%%
# Improved XGBoost approaches for log-normal distribution
def create_xgboost_lognormal_models():
    """
    Create different XGBoost models optimized for log-normal distributions
    """
    
    models = {}
    
    # 1. Standard regression (baseline)
    models['standard'] = {
        'name': 'Standard Regression',
        'params': {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'transform': 'none'
    }
    
    # 2. Log-transformed target
    models['log_transform'] = {
        'name': 'Log-Transformed Target',
        'params': {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'transform': 'log'
    }
    
    # 3. Gamma regression (good for skewed, positive data)
    models['gamma'] = {
        'name': 'Gamma Regression',
        'params': {
            'objective': 'reg:gamma',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'transform': 'none'
    }
    
    # 4. Tweedie regression (handles zeros and skewness)
    models['tweedie'] = {
        'name': 'Tweedie Regression',
        'params': {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.5,  # Between 1 (Poisson) and 2 (Gamma)
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'transform': 'none'
    }
    
    # 5. Quantile regression (focuses on median, robust to outliers)
    models['quantile'] = {
        'name': 'Quantile Regression (50th)',
        'params': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'transform': 'none'
    }
    
    return models

#%%
# Transform functions
def transform_target(y, transform_type='none'):
    """Transform target variable"""
    if transform_type == 'log':
        # Add small constant to handle zeros/negatives
        return np.log(y + 1)
    elif transform_type == 'sqrt':
        # Square root transformation (less aggressive than log)
        return np.sqrt(np.maximum(y, 0))
    else:
        return y

def inverse_transform_target(y_pred, transform_type='none'):
    """Inverse transform predictions back to original scale"""
    if transform_type == 'log':
        return np.exp(y_pred) - 1
    elif transform_type == 'sqrt':
        return y_pred ** 2
    else:
        return y_pred

#%%
# Comprehensive model comparison for log-normal data
def compare_lognormal_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Compare different XGBoost approaches for log-normal distributed targets
    """
    
    models_config = create_xgboost_lognormal_models()
    results = []
    
    print("🔬 COMPARING MODELS FOR LOG-NORMAL DISTRIBUTION")
    print("=" * 60)
    
    for model_key, config in models_config.items():
        print(f"\nTraining {config['name']}...")
        
        try:
            # Transform target if needed
            y_train_transformed = transform_target(y_train, config['transform'])
            y_val_transformed = transform_target(y_val, config['transform'])
            y_test_transformed = transform_target(y_test, config['transform'])
            
            # Create and train model
            model = xgb.XGBRegressor(**config['params'])
            model.fit(X_train, y_train_transformed)
            
            # Make predictions
            y_train_pred_transformed = model.predict(X_train)
            y_val_pred_transformed = model.predict(X_val)
            y_test_pred_transformed = model.predict(X_test)
            
            # Inverse transform predictions
            y_train_pred = inverse_transform_target(y_train_pred_transformed, config['transform'])
            y_val_pred = inverse_transform_target(y_val_pred_transformed, config['transform'])
            y_test_pred = inverse_transform_target(y_test_pred_transformed, config['transform'])
            
            # Calculate metrics on original scale
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Calculate metrics specifically for high scorers (top 20%)
            high_scorer_threshold = np.percentile(y_test, 80)
            high_scorer_mask = y_test >= high_scorer_threshold
            
            if high_scorer_mask.sum() > 0:
                high_scorer_mae = mean_absolute_error(y_test[high_scorer_mask], y_test_pred[high_scorer_mask])
                high_scorer_r2 = r2_score(y_test[high_scorer_mask], y_test_pred[high_scorer_mask])
            else:
                high_scorer_mae = np.nan
                high_scorer_r2 = np.nan
            
            result = {
                'model': config['name'],
                'transform': config['transform'],
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'high_scorer_mae': high_scorer_mae,
                'high_scorer_r2': high_scorer_r2,
                'model_obj': model
            }
            
            results.append(result)
            
            print(f"  Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}, High Scorer R²: {high_scorer_r2:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    
    # Display comparison
    print(f"\n📈 MODEL COMPARISON RESULTS:")
    print("=" * 80)
    print(f"{'Model':<25} {'Test R²':<8} {'Test RMSE':<10} {'Test MAE':<9} {'High R²':<8}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} {row['test_r2']:<8.4f} {row['test_rmse']:<10.3f} {row['test_mae']:<9.3f} {row['high_scorer_r2']:<8.4f}")
    
    # Find best model
    best_overall_idx = results_df['test_r2'].idxmax()
    best_high_scorer_idx = results_df['high_scorer_r2'].idxmax()
    
    print(f"\n🏆 BEST MODELS:")
    print(f"Overall Test R²: {results_df.loc[best_overall_idx, 'model']} ({results_df.loc[best_overall_idx, 'test_r2']:.4f})")
    print(f"High Scorers R²: {results_df.loc[best_high_scorer_idx, 'model']} ({results_df.loc[best_high_scorer_idx, 'high_scorer_r2']:.4f})")
    
    return results_df

#%%
# Advanced: Custom objective function for fantasy football
def create_fantasy_optimized_xgboost():
    """
    Create XGBoost model with custom objective optimized for fantasy football ranking
    """
    
    def fantasy_objective(y_pred, dtrain):
        """
        Custom objective that focuses on ranking accuracy rather than point prediction
        Emphasizes getting the high scorers right
        """
        y_true = dtrain.get_label()
        
        # Calculate residuals
        residual = y_true - y_pred
        
        # Weight residuals based on actual points (higher points = higher weight)
        weights = 1 + np.log1p(np.maximum(y_true, 0))  # log(1 + max(y_true, 0))
        
        # Gradient (derivative of loss w.r.t. prediction)
        grad = -2 * residual * weights
        
        # Hessian (second derivative)
        hess = 2 * weights
        
        return grad, hess
    
    def fantasy_eval_metric(y_pred, dtrain):
        """
        Custom evaluation metric: weighted RMSE with emphasis on high scorers
        """
        y_true = dtrain.get_label()
        weights = 1 + np.log1p(np.maximum(y_true, 0))
        
        weighted_mse = np.average((y_true - y_pred) ** 2, weights=weights)
        weighted_rmse = np.sqrt(weighted_mse)
        
        return 'weighted_rmse', weighted_rmse
    
    return fantasy_objective, fantasy_eval_metric

#%%
# Implementation suggestions for your current model
def improve_current_xgboost_for_lognormal(historical_df, future_fixtures_df, elements_df):
    """
    Practical improvements you can apply to your current XGBoost model
    """
    
    print("🔧 PRACTICAL IMPROVEMENTS FOR LOG-NORMAL DISTRIBUTION")
    print("=" * 60)
    
    # 1. Analyze your current distribution
    print("1. Analyzing your current data distribution...")
    dist_stats = analyze_points_distribution(historical_df, 'total_points')
    
    # 2. Recommend best approach based on distribution
    skewness = dist_stats['skewness']
    
    print(f"\n2. Recommendations based on skewness ({skewness:.3f}):")
    
    if skewness > 2.0:
        print("   📊 High skewness detected!")
        print("   ✅ RECOMMENDED: Log transformation or Gamma/Tweedie objective")
        print("   📝 Try: objective='reg:gamma' or log-transform target")
        recommended_approach = 'log_transform'
    elif skewness > 1.0:
        print("   📊 Moderate skewness detected")
        print("   ✅ RECOMMENDED: Tweedie regression or log transformation")
        print("   📝 Try: objective='reg:tweedie' with variance_power=1.5")
        recommended_approach = 'tweedie'
    else:
        print("   📊 Low skewness - current approach likely fine")
        print("   ✅ RECOMMENDED: Keep current reg:squarederror")
        recommended_approach = 'standard'
    
    # 3. Sample code for improvement
    print(f"\n3. Sample XGBoost parameters for your data:")
    
    if recommended_approach == 'log_transform':
        print("""
# Log transformation approach:
y_train_log = np.log(y_train + 1)  # +1 to handle zeros
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300
)
model.fit(X_train, y_train_log)
predictions_log = model.predict(X_test)
predictions = np.exp(predictions_log) - 1  # Transform back
""")
    
    elif recommended_approach == 'tweedie':
        print("""
# Tweedie regression approach:
model = xgb.XGBRegressor(
    objective='reg:tweedie',
    tweedie_variance_power=1.5,  # Between Poisson (1) and Gamma (2)
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""")
    
    elif recommended_approach == 'gamma':
        print("""
# Gamma regression approach:
model = xgb.XGBRegressor(
    objective='reg:gamma',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300
)
model.fit(X_train, y_train + 0.1)  # Small constant for zeros
predictions = model.predict(X_test)
""")
    
    # 4. Additional tips
    print(f"\n4. Additional tips for fantasy football:")
    print("   🎯 Focus on ranking accuracy over exact point prediction")
    print("   📈 Consider ensemble of multiple approaches")
    print("   🏆 Weight high-scoring players more heavily in loss function")
    print("   📊 Use quantile regression for conservative estimates")
    
    return recommended_approach, dist_stats

#%%
# Quick implementation: Add this to your existing training code
def get_improved_xgboost_params(approach='auto', target_data=None):
    """
    Get improved XGBoost parameters based on target distribution
    
    Args:
        approach: 'auto', 'log_transform', 'tweedie', 'gamma', or 'standard'
        target_data: Target variable data for auto-detection
    
    Returns:
        dict: XGBoost parameters and preprocessing info
    """
    
    if approach == 'auto' and target_data is not None:
        skewness = stats.skew(target_data.dropna())
        if skewness > 2.0:
            approach = 'log_transform'
        elif skewness > 1.0:
            approach = 'tweedie'
        else:
            approach = 'standard'
        
        print(f"🤖 Auto-detected approach: {approach} (skewness: {skewness:.3f})")
    
    configs = {
        'standard': {
            'params': {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'transform': None,
            'description': 'Standard squared error regression'
        },
        
        'log_transform': {
            'params': {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'transform': 'log',
            'description': 'Log-transformed target with squared error'
        },
        
        'tweedie': {
            'params': {
                'objective': 'reg:tweedie',
                'tweedie_variance_power': 1.5,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'transform': None,
            'description': 'Tweedie regression (handles zeros and skewness)'
        },
        
        'gamma': {
            'params': {
                'objective': 'reg:gamma',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'transform': 'add_small_constant',
            'description': 'Gamma regression (good for positive skewed data)'
        }
    }
    
    return configs[approach]

# Example usage:
print("💡 TO IMPROVE YOUR CURRENT MODEL:")
print("=" * 40)
print("1. Add this to your XGBoost training code:")
print("   config = get_improved_xgboost_params('auto', y_train)")
print("   model = xgb.XGBRegressor(**config['params'])")
print("")
print("2. Or manually try Tweedie regression:")
print("   model = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5)")
print("")
print("3. For log transformation:")
print("   y_train_log = np.log(y_train + 1)")
print("   # Train on y_train_log, then: predictions = np.exp(pred) - 1")
#%%