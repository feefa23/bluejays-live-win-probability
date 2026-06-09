import requests
import json

# Using our completed historical game ID (Yankees vs. Blue Jays)
GAME_ID = 744910

print(f"Fetching live data directly from MLB servers for Game ID {GAME_ID}...")

# This is the exact raw web link MLB uses for its mobile apps and websites
url = f"https://statsapi.mlb.com/api/v1.1/game/{GAME_ID}/feed/live"

# Hit the web link and get the data
response = requests.get(url)

# Convert the raw text from the internet into a clean Python dictionary
live_data = response.json()

# Dig into the data folders safely
if 'liveData' in live_data and 'linescore' in live_data['liveData']:
    linescore = live_data['liveData']['linescore']
    plays = live_data['liveData']['plays']
    
    # Try to grab the ball and strike count from the current play
    current_play = plays.get('currentPlay', {})
    count = current_play.get('count', {})
    
    # Extract the exact numbers our Machine Learning model needs
    baseball_state = {
        "inning": linescore.get("currentInning"),
        "half_inning": linescore.get("inningHalf"), # Top or Bottom
        "outs": linescore.get("outs"),
        "balls": count.get('balls', 0),
        "strikes": count.get('strikes', 0),
        "home_score": linescore.get("teams", {}).get("home", {}).get("runs", 0),
        "away_score": linescore.get("teams", {}).get("away", {}).get("runs", 0),
    }
    
    # Calculate score differential (Home Team Score - Away Team Score)
    baseball_state["score_differential"] = baseball_state["home_score"] - baseball_state["away_score"]
    
    print("\n--- Map of current ML Feature Vector ---")
    print(json.dumps(baseball_state, indent=4))
    
else:
    print("\nCould not find live play data inside the MLB server response.")