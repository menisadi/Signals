import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read in cellular data and antenna data
signals_data = pd.read_csv('signals_data.csv')

# Function to calculate the distance between a signal and an antenna
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a)) # 2*R*asin...

# Function to find the closest antenna to a signal
def closest_antenna(signal_lat, signal_lon):
    closest_distance = float('inf')
    closest_antenna = None
    for i, row in antennas_data.iterrows():
        antenna_lat = row['latitude']
        antenna_lon = row['longitude']
        d = distance(signal_lat, signal_lon, antenna_lat, antenna_lon)
        if d < closest_distance:
            closest_distance = d
            closest_antenna = row['name']
    return closest_antenna, closest_distance

# Add the closest antenna and distance columns to the signals dataframe
signals_data['closest_antenna'], signals_data['distance'] = zip(*signals_data.apply(lambda x: closest_antenna(x['latitude'], x['longitude']), axis=1))

# Split the data into features and target
X = signals_data[['closest_antenna', 'distance']]
y = signals_data['signal_strength']

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['closest_antenna'])
    ])

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
test_score = pipeline.score(X_test, y_test)
print("Test score: ", test_score)
