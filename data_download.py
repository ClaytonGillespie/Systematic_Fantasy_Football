import os
import requests
import pandas as pd
from time import sleep

# URLs
BASE_DATA_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
PLAYER_DETAIL_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"

# Output folder
DATA_DIR = "./data/"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_base_data():
    response = requests.get(BASE_DATA_URL)
    response.raise_for_status()
    data = response.json()

    players = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    positions = pd.DataFrame(data['element_types'])
    events = pd.DataFrame(data['events'])

    # Determine current gameweek
    current_gw = events[events['is_current'] == True]
    gameweek = int(current_gw.iloc[0]['id']) if not current_gw.empty else 0

    return players, teams, positions, events, gameweek

def enhance_players(players, teams, positions):
    players = players.merge(
        teams[['id', 'name', 'short_name']],
        left_on='team',
        right_on='id',
        suffixes=('', '_team')
    ).rename(columns={'name': 'team_name', 'short_name': 'team_short'})

    players = players.merge(
        positions[['id', 'singular_name']],
        left_on='element_type',
        right_on='id',
        suffixes=('', '_position')
    ).rename(columns={'singular_name': 'position'})

    players.drop(columns=['id_team', 'id_position'], errors='ignore', inplace=True)
    return players

def fetch_fixtures():
    response = requests.get(FIXTURES_URL)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def fetch_player_history(player_ids, rate_limit=0.5):
    all_history = []
    all_upcoming = []

    for pid in player_ids:
        url = PLAYER_DETAIL_URL.format(pid)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch player {pid}")
            continue
        data = response.json()

        history = pd.DataFrame(data['history'])
        fixtures = pd.DataFrame(data['fixtures'])

        if not history.empty:
            history['player_id'] = pid
            all_history.append(history)

        if not fixtures.empty:
            fixtures['player_id'] = pid
            all_upcoming.append(fixtures)

        sleep(rate_limit)

    history_df = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    upcoming_df = pd.concat(all_upcoming, ignore_index=True) if all_upcoming else pd.DataFrame()

    return history_df, upcoming_df

def save_to_parquet(df: pd.DataFrame, name: str, gameweek: int):
    filename = f"{DATA_DIR}{name}_gw{gameweek}.parquet"
    df.to_parquet(filename, index=False)
    print(f"Saved: {filename}")

def main():
    players, teams, positions, events, gameweek = fetch_base_data()
    players = enhance_players(players, teams, positions)
    fixtures = fetch_fixtures()

    player_ids = players['id'].tolist()
    player_history, player_upcoming = fetch_player_history(player_ids)

    players_summary = players[[
        'id', 'first_name', 'second_name', 'team_name', 'team_short', 'position',
        'now_cost', 'total_points', 'selected_by_percent', 'form', 'minutes'
    ]]

    # Save all DataFrames
    save_to_parquet(players_summary, "players", gameweek)
    save_to_parquet(teams, "teams", gameweek)
    save_to_parquet(positions, "positions", gameweek)
    save_to_parquet(events, "events", gameweek)
    save_to_parquet(fixtures, "fixtures", gameweek)
    save_to_parquet(player_history, "player_history", gameweek)
    save_to_parquet(player_upcoming, "player_upcoming_fixtures", gameweek)

if __name__ == "__main__":
    main()
