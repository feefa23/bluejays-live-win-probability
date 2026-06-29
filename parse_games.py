import requests
import json
import csv
import time

output_file = "blue_jays_training_data.csv"

headers = [
    "game_id", "inning", "is_blue_jays_batting", "outs", "balls", "strikes",
    "runner_on_1st", "runner_on_2nd", "runner_on_3rd", "score_differential",
    "batter_avg", "batter_ops", "pitcher_era", "pitcher_so_rate", "blue_jays_won"
]

print("Initializing On-Demand Lazy Loading Processing Engine...")
batter_cache = {}
pitcher_cache = {}

def get_batter_stats(b_id):
    if b_id in batter_cache:
        return batter_cache[b_id]
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{b_id}/stats?stats=season&season=2026&group=hitting"
        res = requests.get(url).json().get('stats', [])
        if res:
            stat = res[0].get('splits', [{}])[0].get('stat', {})
            batter_cache[b_id] = {
                'avg': float(stat.get('avg', '.000').replace('.', '0.') if '.' in str(stat.get('avg')) else 0.245),
                'ops': float(stat.get('ops', 0.730))
            }
            return batter_cache[b_id]
    except Exception:
        pass
    return {'avg': 0.245, 'ops': 0.730}

def get_pitcher_stats(p_id):
    if p_id in pitcher_cache:
        return pitcher_cache[p_id]
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{p_id}/stats?stats=season&season=2026&group=pitching"
        res = requests.get(url).json().get('stats', [])
        if res:
            stat = res[0].get('splits', [{}])[0].get('stat', {})
            raw_era = stat.get('era')
            era_val = float(raw_era) if (raw_era and raw_era != '-.--') else 3.95
            outs = float(stat.get('outs', 1))
            so = float(stat.get('strikeOuts', 0))
            pitcher_cache[p_id] = {
                'era': era_val,
                'so_rate': float(so / outs if outs > 0 else 0.22)
            }
            return pitcher_cache[p_id]
    except Exception:
        pass
    return {'era': 3.95, 'so_rate': 0.22}

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

with open("blue_jays_ids_2025.txt", "r") as f:
    game_ids = f.read().splitlines()

for index, game_id in enumerate(game_ids):
    print(f"[{index + 1}/{len(game_ids)}] Processing historical context rows for Game ID: {game_id}...")
    
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        response = requests.get(url)
        live_data = response.json()
        
        live_data_folder = live_data.get('liveData', {})
        linescore = live_data_folder.get('linescore', {})
        home_team_id = live_data.get('gameData', {}).get('teams', {}).get('home', {}).get('id')
        
        home_runs = linescore.get('teams', {}).get('home', {}).get('runs', 0)
        away_runs = linescore.get('teams', {}).get('away', {}).get('runs', 0)
        
        bj_won = 0
        is_home = True if home_team_id == 141 else False
        if is_home and home_runs > away_runs: bj_won = 1
        elif not is_home and away_runs > home_runs: bj_won = 1

        all_plays = live_data_folder.get('plays', {}).get('allPlays', [])
        
        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            for play in all_plays:
                about = play.get('about', {})
                count = play.get('count', {})
                result = play.get('result', {})
                matchup = play.get('matchup', {})
                
                inning = about.get('inning', 1)
                half_inning = about.get('halfInning')
                outs = count.get('outs', 0)
                balls = count.get('balls', 0)
                strikes = count.get('strikes', 0)
                
                is_bj_batting = 1 if (half_inning == 'bottom' and is_home) or (half_inning == 'top' and not is_home) else 0
                runner_on_1st = 1 if 'postOnFirst' in matchup else 0
                runner_on_2nd = 1 if 'postOnSecond' in matchup else 0
                runner_on_3rd = 1 if 'postOnThird' in matchup else 0
                
                home_score = result.get('homeScore', 0)
                away_score = result.get('awayScore', 0)
                score_differential = home_score - away_score
                
                b_id = str(matchup.get('batter', {}).get('id'))
                p_id = str(matchup.get('pitcher', {}).get('id'))
                
                b_stats = get_batter_stats(b_id)
                p_stats = get_pitcher_stats(p_id)
                
                row = [
                    game_id, inning, is_bj_batting, outs, balls, strikes,
                    runner_on_1st, runner_on_2nd, runner_on_3rd, score_differential,
                    b_stats['avg'], b_stats['ops'], p_stats['era'], p_stats['so_rate'],
                    bj_won
                ]
                writer.writerow(row)
        time.sleep(0.01)
    except Exception as e:
        continue

print("\nSuccess! Custom dataset built with full dynamic player registries.")