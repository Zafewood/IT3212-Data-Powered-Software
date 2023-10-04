import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('energy_dataset.csv')

print(data['time'])
plt.plot(data['time'], data['generation biomass'])
plt.show()