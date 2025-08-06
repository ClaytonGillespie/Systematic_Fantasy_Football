
# Fantasy Football Optimization Suite

A comprehensive Fantasy Premier League (FPL) optimization tool that uses machine learning and linear programming to predict player performance and generate optimal team selections.

## Overview

This project combines historical FPL data analysis, XGBoost machine learning models, and PuLP optimization to:
- Predict player performance for upcoming gameweeks
- Generate optimal team selections for single and multi-period strategies
- Scrape expert recommendations from Fantasy Football Scout
- Optimize transfers and captaincy decisions

## Features

- **Machine Learning Predictions**: XGBoost model trained on historical player performance data
- **Team Optimization**: Single and multi-period FPL team optimization using linear programming
- **Expert Integration**: Fantasy Football Scout scraper for incorporating expert recommendations
- **Historical Data**: Comprehensive dataset covering 2023-2025 seasons
- **Transfer Planning**: Multi-gameweek transfer optimization

## Project Structure

- `model.py` - XGBoost model training and prediction generation
- `fpl_optimizer.py` - Basic FPL optimization logic
- `fpl_optimizer_single_period.py` - Single gameweek optimization
- `fpl_optimizer_multi_period.py` - Multi-gameweek optimization with transfers
- `ffs_scraper.py` - Fantasy Football Scout recommendations scraper
- `data_download.py` - FPL API data collection script
- `data_2023/`, `data_2024/`, `data_2025/` - Historical and current season data

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Fantasy_Football_v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

## Usage

### Generate Predictions
```bash
python model.py
```

### Optimize Team Selection
```bash
# Single gameweek optimization
python fpl_optimizer_single_period.py

# Multi-gameweek optimization with transfers
python fpl_optimizer_multi_period.py
```

### Scrape Expert Recommendations
```bash
python ffs_scraper.py
```

### Download Latest Data
```bash
python data_download.py
```

## Data Sources

- **FPL API**: Official Fantasy Premier League API for player stats and fixtures
- **Fantasy Football Scout**: Expert recommendations and insights
- **Historical Data**: Multi-season player performance and team data

## Model Performance

The XGBoost model uses features including:
- Player historical performance
- Fixture difficulty ratings
- Team form and statistics
- Previous season performance

## Output Files

Optimized teams and predictions are saved to:
- `data_2025/predictions/` - Model predictions and optimal teams
- `data_2025/ffs_recommendations/` - Expert recommendations

## Dependencies

Key libraries used:
- `xgboost` - Machine learning model
- `pulp` - Linear programming optimization
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Model evaluation
- `beautifulsoup4` - Web scraping
- `requests` - API calls

## Future Thoughts

### Model Export and Sharing
- Build XGBoost model based on previous years
- Think about pulp scoring: captain, bench and transfers

### Considerations
- Dealing with changing team ids (relegations)
- Player transfers
- Free to change captain
- How many bench positions to actually use