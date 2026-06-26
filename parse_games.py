import requests
import json
import csv
import time

output_file = "blue_jays_training_data.csv"

# Our feature layout has expanded to include historical player performance context!
headers = [
    "game_id", "inning", "is_blue_jays_batting", "outs", "balls", "strikes",
    "runner_on_1st", "runner_on_2nd", "runner_on_3rd", "score_differential",
    "batter_avg", "batter_ops", "pitcher_era", "pitcher_so_rate", "blue_jays_won"
]

print("Initializing Master Player-Aware Baseball Dataset...")

# 1. PRE-FETCH PLAYER STATS CACHE FOR THE 2025 SEASON
print("Caching 2025 player performance registries from MLB API...")
batter_cache = {}
pitcher_cache = {}

try:
    # Fetch all hitting stats for 2025 season
    b_url = "https://statsapi.mlb.com/api/v1/stats?stats=season&season=2025&group=hitting&limit=1500"
    b_data = requests.get(b_url).json().get('stats', [{}])[0].get('splits', [])
    for split in b_data:
        p_id = split.get('player', {}).get('id')
        stats = split.get('stat', {})
        batter_cache[p_id] = {
            'avg': float(stats.get('avg', '.000').replace('.', '0.') if '.' in str(stats.get('avg')) else 0.0),
            'ops': float(stats.get('ops', 0.0))
        }
        
    # Fetch all pitching stats for 2025 season
    p_url = "https://statsapi.mlb.com/api/v1/stats?stats=season&season=2025&group=pitching&limit=1500"
    p_data = requests.get(p_url).json().get('stats', [{}])[0].get('splits', [])
    for split in p_data:
        p_id = split.get('player', {}).get('id')
        stats = split.get('stat', {})
        outs = float(stats.get('outs', 1))
        so = float(stats.get('strikeOuts', 0))
        pitcher_cache[p_id] = {
            'era': float(stats.get('era', 4.50) if stats.get('era') != '-.--' else 4.50),
            'so_rate': float(so / outs if outs > 0 else 0.0) # Proxy for Whiff / SO tracking efficiency
        }
    print(f"Successfully cached data for {len(batter_cache)} batters and {len(pitcher_cache)} pitchers.\n")
except Exception as e:
    print(f"Warning, stats cache initialization encountered an anomaly: {e}. Falling back to default baseline metrics.")

# 2. GENERATE SPREADSHEET HEADERS
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

with open("blue_jays_ids_2025.txt", "r") as f:
    game_ids = f.read().splitlines()

print(f"Processing play-by-play streams for {len(game_ids)} games...")

for index, game_id in enumerate(game_ids):
    if (index + 1) % 15 == 0 or index == 0:
        print(f"[{index + 1}/{len(game_ids)}] Merging metrics for Game ID: {game_id}...")
    
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    
    try:
        response = requests.get(url)
        live_data = response.json()
        
        live_data_folder = live_data.get('liveData', {})
        linescore = live_data_folder.get('linescore', {})
        home_team_id = live_data.get('gameData', {}).get('teams', {}).get('home', {}).get('id')
        away_team_id = live_data.get('gameData', {}).get('teams', {}).get('away', {}).get('id')
        
        home_runs = linescore.get('teams', {}).get('home', {}).get('runs', 0)
        away_runs = linescore.get('teams', {}).get('away', {}).get('runs', 0)
        
        bj_won = 0
        is_home = False
        if home_team_id == 141:
            is_home = True
            if home_runs > away_runs: bj_won = 1
        elif away_team_id == 141:
            if away_runs > home_runs: bj_won = 1

        all_plays = live_data_folder.get('plays', {}).get('allPlays', [])
        
        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            
            for play in all_plays:
                about = play.get('about', {})
                count = play.get('count', {})
                result = play.get('result', {})
                matchup = play.get('matchup', {})
                
                inning = about.get('inning')
                half_inning = about.get('halfInning')
                outs = count.get('outs', 0)
                balls = count.get('balls', 0)
                strikes = count.get('strikes', 0)
                
                is_bj_batting = 0
                if (half_inning == 'bottom' and is_home) or (half_inning == 'top' and not is_home):
                    is_bj_batting = 1
                
                runner_on_1st = 1 if 'postOnFirst' in matchup else 0
                runner_on_2nd = 1 if 'postOnSecond' in matchup else 0
                runner_on_3rd = 1 if 'postOnThird' in matchup else 0
                
                home_score = result.get('homeScore', 0)
                away_score = result.get('awayScore', 0)
                score_differential = home_score - away_score
                
                # Extract player identifiers
                b_id = matchup.get('batter', {}).get('id')
                p_id = matchup.get('pitcher', {}).get('id')
                
                # Fetch statistics from our local cache dictionary (or use league averages if not found)
                b_stats = batter_cache.get(b_id, {'avg': 0.245, 'ops': 0.730})
                p_stats = pitcher_cache.get(p_id, {'era': 4.20, 'so_rate': 0.22})
                
                row = [
                    game_id, inning, is_bj_batting, outs, balls, strikes,
                    runner_on_1st, runner_on_2nd, runner_on_3rd, score_differential,
                    b_stats['avg'], b_stats['ops'], p_stats['era'], p_stats['so_rate'],
                    bj_won
                ]
                writer.writerow(row)
                
        time.sleep(0.05)
        
    except Exception as e:
        continue

print("\nSuccess! Custom dataset built with full batter and pitcher situational data.")