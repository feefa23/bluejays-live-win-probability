import requests
import json
import csv
import time

# 1. Setup the master spreadsheet file
output_file = "blue_jays_training_data.csv"

# These are the column names for our master spreadsheet
headers = [
    "game_id", "inning", "half_inning", "outs", 
    "balls", "strikes", "home_score", "away_score", 
    "score_differential", "blue_jays_won"
]

print("Initializing your master baseball dataset...")

# Create the CSV file and write the top header row
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# 2. Read our saved game IDs from the text file
print("Reading saved game IDs...")
with open("blue_jays_ids_2025.txt", "r") as f:
    # .read().splitlines() reads the file line-by-line and removes extra spaces
    game_ids = f.read().splitlines()

print(f"Loaded {len(game_ids)} games to process. Starting download loop...\n")

# 3. Loop through every single game ID
# To keep this fast for your testing, we will just scrape the first 5 games for now!
for index, game_id in enumerate(game_ids[:5]):
    print(f"[{index + 1}/5] Processing Game ID: {game_id}...")
    
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    
    try:
        response = requests.get(url)
        live_data = response.json()
        
        # Figure out who won this game ahead of time
        # Team 141 is the Blue Jays
        live_data_folder = live_data.get('liveData', {})
        linescore = live_data_folder.get('linescore', {})
        
        home_team_id = live_data.get('gameData', {}).get('teams', {}).get('home', {}).get('id')
        away_team_id = live_data.get('gameData', {}).get('teams', {}).get('away', {}).get('id')
        
        home_runs = linescore.get('teams', {}).get('home', {}).get('runs', 0)
        away_runs = linescore.get('teams', {}).get('away', {}).get('runs', 0)
        
        # Determine if the Blue Jays won this specific game
        bj_won = 0
        if home_team_id == 141 and home_runs > away_runs:
            bj_won = 1
        elif away_team_id == 141 and away_runs > home_runs:
            bj_won = 1

        # Dig into the sequence of actual plays
        all_plays = live_data_folder.get('plays', {}).get('allPlays', [])
        
        # We will open our CSV file in "append" mode ('a') to add rows to the bottom
        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            
            # Loop through every play that happened in this single game
            for play in all_plays:
                about = play.get('about', {})
                count = play.get('count', {})
                result = play.get('result', {})
                
                # Extract the variables
                inning = about.get('inning')
                half_inning = about.get('halfInning') # Top or Bottom
                outs = count.get('outs', 0)
                balls = count.get('balls', 0)
                strikes = count.get('strikes', 0)
                
                # Scores *at the moment of this specific play*
                home_score = result.get('homeScore', 0)
                away_score = result.get('awayScore', 0)
                score_differential = home_score - away_score
                
                # Package all 10 items into a clean row list
                row = [
                    game_id, inning, half_inning, outs, 
                    balls, strikes, home_score, away_score, 
                    score_differential, bj_won
                ]
                
                # Write this specific moment into our spreadsheet
                writer.writerow(row)
                
        # Pause for half a second so we don't spam the MLB servers too hard
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error processing game {game_id}: {e}")
        continue

print("\nSuccess! Your starter dataset has been created.")