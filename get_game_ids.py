import statsapi

# Scrape all games from the 2025 regular season
YEAR = 2025
BLUE_JAYS_ID = 141

print(f"Searching for all Blue Jays games in the {YEAR} season...")

# Fetch the entire schedule for the Blue Jays for that specific year
# Specify the regular season ('R') so we don't accidentally get spring training games
games = statsapi.schedule(team=BLUE_JAYS_ID, start_date=f"{YEAR}-03-01", end_date=f"{YEAR}-10-31")

game_ids = []

# Loop through every single game found in that time frame
for game in games:
    # Double check that it's a regular season game and has a valid ID
    if game.get('game_type') == 'R':
        gid = game.get('game_id')
        if gid:
            game_ids.append(str(gid))

print(f"Success! Found {len(game_ids)} regular season games for the Blue Jays in {YEAR}.")

# Save all these IDs into a clean text file so we can use them later
output_filename = f"blue_jays_ids_{YEAR}.txt"
with open(output_filename, "w") as f:
    f.write("\n".join(game_ids))

print(f"Saved all game IDs to {output_filename}")