import requests
import torch
import torch.nn as nn
import statsapi
import numpy as np
import time
from datetime import datetime, timedelta

class WinProbabilityModel(nn.Module):
    def __init__(self):
        super(WinProbabilityModel, self).__init__()
        self.hidden = nn.Linear(13, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = WinProbabilityModel()
model.load_state_dict(torch.load("blue_jays_model.pth"))
model.eval()

BLUE_JAYS_ID = 141

feature_mins = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
feature_maxs = np.array([12.0, 1.0, 3.0, 4.0, 3.0, 1.0, 1.0, 1.0, 15.0, 0.400, 1.200, 15.0, 1.0], dtype=np.float32)

print("Initializing live matchup tracking engine with target player tracking lookups...")
batter_cache = {}
pitcher_cache = {}

def get_live_batter_stats(b_id):
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

def get_live_pitcher_stats(p_id):
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

def get_active_game_id():
    current_date = datetime.now()
    for i in range(5):
        date_str = current_date.strftime("%Y-%m-%d")
        schedule = statsapi.schedule(team=BLUE_JAYS_ID, date=date_str)
        if schedule:
            game = schedule[0]
            status = game.get('status', '')
            game_id = game.get('game_id')
            if status in ['In Progress', 'Final', 'Completed', 'Game Over']:
                return game_id
        current_date -= timedelta(days=1)
    return None

GAME_ID = get_active_game_id()
if not GAME_ID:
    print("Could not locate any recent matches.")
    exit()

url = f"https://statsapi.mlb.com/api/v1.1/game/{GAME_ID}/feed/live"

try:
    response = requests.get(url)
    live_data = response.json()
    
    away_team = live_data.get('gameData', {}).get('teams', {}).get('away', {}).get('name')
    home_team = live_data.get('gameData', {}).get('teams', {}).get('home', {}).get('name')
    home_team_id = live_data.get('gameData', {}).get('teams', {}).get('home', {}).get('id')
    is_home = True if home_team_id == BLUE_JAYS_ID else False
    
    print(f"\nTracking Game ID: {GAME_ID}")
    print(f"Matchup: {away_team} @ {home_team}\n")
    print("==========================================================================================")
    
    all_plays = live_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
    
    for index, play in enumerate(all_plays):
        about = play.get('about', {})
        count = play.get('count', {})
        result = play.get('result', {})
        matchup = play.get('matchup', {})
        
        inning = about.get('inning', 1)
        half_inning = about.get('halfInning', 'top')
        outs = count.get('outs', 0)
        balls = count.get('balls', 0)
        strikes = count.get('strikes', 0)
        
        is_bj_batting = 1 if (half_inning == 'bottom' and is_home) or (half_inning == 'top' and not is_home) else 0
        runner_on_1st = 1 if 'postOnFirst' in matchup else 0
        runner_on_2nd = 1 if 'postOnSecond' in matchup else 0
        runner_on_3rd = 1 if 'postOnThird' in matchup else 0
        
        home_score = result.get('homeScore', 0)
        away_score = result.get('awayScore', 0)
        
        bj_score = home_score if is_home else away_score
        opp_score = away_score if is_home else home_score
        score_differential = home_score - away_score
        
        batter_name = matchup.get('batter', {}).get('fullName', 'Unknown Batter')
        pitcher_name = matchup.get('pitcher', {}).get('fullName', 'Unknown Pitcher')
        
        b_id = str(matchup.get('batter', {}).get('id'))
        p_id = str(matchup.get('pitcher', {}).get('id'))
        
        b_stats = get_live_batter_stats(b_id)
        p_stats = get_live_pitcher_stats(p_id)
        
        # FIXED: Precise, non-leaking boundary check condition loops
        is_game_over = False
        if inning >= 9:
            if half_inning == 'top' and outs == 3 and bj_score > opp_score:
                is_game_over = True
                win_probability = 100.00
            elif half_inning == 'bottom' and outs == 3:
                is_game_over = True
                win_probability = 100.00 if bj_score > opp_score else 0.00
            elif half_inning == 'bottom' and bj_score > opp_score:
                is_game_over = True
                win_probability = 100.00

        if not is_game_over:
            current_moment = np.array([
                float(inning), float(is_bj_batting), float(outs), float(balls), float(strikes),
                float(runner_on_1st), float(runner_on_2nd), float(runner_on_3rd), float(score_differential),
                b_stats['avg'], b_stats['ops'], p_stats['era'], p_stats['so_rate']
            ], dtype=np.float32)
            
            scaled_moment = (current_moment - feature_mins) / (feature_maxs - feature_mins)
            input_tensor = torch.from_numpy(scaled_moment).float().unsqueeze(0)
            
            with torch.no_grad():
                win_probability = model(input_tensor).item() * 100

        bases_occupied = []
        if runner_on_1st: bases_occupied.append("1st")
        if runner_on_2nd: bases_occupied.append("2nd(RISP)")
        if runner_on_3rd: bases_occupied.append("3rd")
        bases_str = ", ".join(bases_occupied) if bases_occupied else "Empty"
        
        print(f"Play {index + 1}/{len(all_plays)} | {half_inning.upper()} {inning} | Outs: {outs} | Count: {balls}-{strikes}")
        print(f"Score   : {away_team} {away_score} vs {home_team} {home_score}")
        print(f"Bases   : [{bases_str}]")
        print(f"Matchup : Batter: {batter_name} (AVG: {b_stats['avg']:.3f}, OPS: {b_stats['ops']:.3f})")
        print(f"          Pitcher: {pitcher_name} (ERA: {p_stats['era']:.2f})")
        print(f">>> Blue Jays Live Win Probability: {win_probability:.2f}%\n")
        print("-" * 90)
        
        time.sleep(0.1)

except Exception as e:
    print(f"Error tracking live game: {e}")