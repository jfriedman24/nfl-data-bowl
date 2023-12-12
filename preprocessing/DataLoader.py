import pandas as pd

# Function that loads data in order [games_df, players_df, plays_df, tracking_df]
def load_data():
    frames = []

    frame_names = ['games', 'players', 'plays']
    for name in frame_names:
        filename = "../data/" + name + ".csv"
        df = pd.read_csv(filename)
        frames += [df]
        print("loaded " + name + " df")
        print("shape: " + str(df.shape)) 
        print("-----")
    
    print("loading tracking frames...")
    # Load each week's data and keep track in array
    tracking_frames = []
    for week in range(1,10):
        filename = "../data/tracking_week_" + str(week) + ".csv"
        df = pd.read_csv(filename)
        tracking_frames += [df]

    # Combine into 1 frame
    tracking_df = pd.concat(tracking_frames)

    # Add to list
    frames += [tracking_df]
    print("loaded tracking frames")
    print("shape: " + str(tracking_df.shape))
    print("returning " + str(len(frames)) + " frames")
    return frames

def load_data_tubevit():
    frames = load_data()

    print("loading 2020 data")
    filename = "../data/tracking_data_2020.csv"
    df = pd.read_csv(filename, dtype={'WindSpeed': 'object'})
    print("shape: " + str(df.shape))

    frames += [df]
    return frames
