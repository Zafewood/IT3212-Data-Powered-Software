import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Preperation

weather_df = pd.read_csv('data/weather_features.csv')
weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])

valencia_df = weather_df[weather_df['city_name'] == 'Valencia']
madrid_df = weather_df[weather_df['city_name'] == 'Madrid']
bilbao_df = weather_df[weather_df['city_name'] == 'Bilbao']
barcelona_df = weather_df[weather_df['city_name'] == 'Barcelona']
seville_df = weather_df[weather_df['city_name'] == 'Seville']

