#%%

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

    total = len(player_ids)
    failures = []

    print(f"\n📊 Fetching player match history for {total} players...\n")

    for idx, pid in enumerate(player_ids, 1):
        print(f"⏳ [{idx}/{total}] Fetching player {pid}...", end=' ', flush=True)
        url = PLAYER_DETAIL_URL.format(pid)

        try:
            response = requests.get(url)
            response.raise_for_status()

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                print("❌ JSON decode failed.")
                failures.append(pid)
                continue

            history = pd.DataFrame(data.get('history', []))
            fixtures = pd.DataFrame(data.get('fixtures', []))

            if not history.empty:
                history['player_id'] = pid
                all_history.append(history)

            if not fixtures.empty:
                fixtures['player_id'] = pid
                all_upcoming.append(fixtures)

            print("✅ Success.")

        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            failures.append(pid)
            continue

        sleep(rate_limit)

    history_df = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    upcoming_df = pd.concat(all_upcoming, ignore_index=True) if all_upcoming else pd.DataFrame()

    print(f"\n✅ Done fetching player data.")
    print(f"🎯 Successful: {total - len(failures)} / {total}")
    print(f"⚠️ Failed: {len(failures)} players.")

    return history_df, upcoming_df

def save_to_parquet(df: pd.DataFrame, name: str, gameweek: int):
    filename = f"{DATA_DIR}{name}_gw{gameweek}.parquet"
    df.to_parquet(filename, index=False)
    print(f"Saved: {filename}")

# Assuming `events` is your DataFrame and overrides is a column
def unpack_overrides(events_df):
    def extract_override(row):
        override = row.get('overrides')
        if isinstance(override, list) and len(override) > 0:
            return override[0]
        return {}

    # Apply extraction to each row
    overrides_expanded = events_df['overrides'].apply(extract_override).apply(pd.Series)

    # Rename override keys to prefixed columns
    overrides_expanded = overrides_expanded.rename(columns={
        'rules': 'override_rules',
        'scoring': 'override_scoring',
        'element_types': 'override_element_types',
        'pick_multiplier': 'override_pick_multiplier'
    })

    # Drop original overrides column and join the new columns
    events_flat = pd.concat([events_df.drop(columns=['overrides']), overrides_expanded], axis=1)

    return events_flat

#%%

players, teams, positions, events, gameweek = fetch_base_data()
players = enhance_players(players, teams, positions)
fixtures = fetch_fixtures()

player_ids = players['id'].tolist()
player_history, player_upcoming = fetch_player_history(player_ids)

players_summary = players[[
    'id', 'first_name', 'second_name', 'team_name', 'team_short', 'position',
    'now_cost', 'total_points', 'selected_by_percent', 'form', 'minutes'
]]

# Apply function
events_flat = unpack_overrides(events)

# Save all DataFrames
save_to_parquet(players_summary, "players", gameweek)
save_to_parquet(teams, "teams", gameweek)
save_to_parquet(positions, "positions", gameweek)
save_to_parquet(events_flat, "events", gameweek)
save_to_parquet(fixtures, "fixtures", gameweek)
save_to_parquet(player_history, "player_history", gameweek)
save_to_parquet(player_upcoming, "player_upcoming_fixtures", gameweek)

#%%