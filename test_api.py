import statsapi

print("Searching for the Toronto Blue Jays team data...")

# 1. Look up the Blue Jays team information
teams = statsapi.lookup_team('toronto')

if teams:
    blue_jays = teams[0]
    bj_id = blue_jays['id']
    print(f"Success! Found Team: {blue_jays['name']} (ID: {bj_id})")
    
    print("\nFetching the current or most recent game for the Blue Jays...")
    
    # 2. Get the most recent game schedule data for the Blue Jays
    schedule = statsapi.schedule(team=bj_id)
    
    if schedule:
        latest_game = schedule[0]
        game_id = latest_game['game_id']
        print(f"Found Game ID: {game_id}")
        print(f"Matchup: {latest_game['away_name']} @ {latest_game['home_name']}")
        print(f"Status: {latest_game['status']}")
    else:
        print("No game found for today. The API connection works perfectly though!")
else:
    print("Could not find the Blue Jays. Check your internet connection!")