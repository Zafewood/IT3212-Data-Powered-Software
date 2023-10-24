from scipy.fft import fft, fftfreq
import scipy
# Number of samples in normalized_tone
# df1 = pd.read_csv('data/preprocessed_energy_dataset.csv')

# df1['generation solar normalized'] = (df1['generation solar'] - df1['generation solar'].min()) / \
#                                     (df1['generation solar'].max() - df1['generation solar'].min())


# N = len(df1['generation solar normalized'])
# yf = fft(df1['generation solar normalized'])
# xf = fftfreq(N)  # You might want to adjust the sampling rate as an argument here
# plt.plot(xf[:N//2], np.abs(yf)[:N//2])  # Displaying only the positive frequencies
# plt.show()
df1 = pd.read_csv('data/preprocessed_energy_dataset.csv')

time_series = np.array(df1['generation solar'])

N = len(time_series)

T = 1.0

# Apply Fourier Transform
fourier_transform = fft(time_series)

# Get absolute values to represent amplitudes
amplitudes = np.abs(fourier_transform)

# Create a frequency axis
frequencies = fftfreq(N, T)

# Only keep the positive frequencies (since the negative frequencies are symmetrical for real-valued signals)
positive_freq_indices = np.where(frequencies > 0)
positive_frequencies = frequencies[positive_freq_indices]
positive_amplitudes = amplitudes[positive_freq_indices]

# Plot the Fourier Transform
plt.figure(figsize=(10, 5))
plt.plot(positive_frequencies, positive_amplitudes)
plt.title('Fourier Transform')
plt.xlabel('Frequency (cycles per unit time)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

filtered = lowpass(time_series, 0.4, 1.0)
inverse_transform = scipy.ifft(fourier_transform)
jalla = np.real(inverse_transform)

# plt.plot(jalla)

decom = seasonal_decompose(jalla, model='additive', period=744)
plt.plot(decom.trend)

# plt.show()
# yf = fft(df1['generation solar'])
# xf = fftfreq(df1['time'])


import tsfel

# preprocessed_energy_df = preprocessed_energy_df.drop(['time'], axis=1)
cfg_file = tsfel.get_features_by_domain('statistical')
X_train = tsfel.time_series_features_extractor(cfg_file, preprocessed_energy_df, fs=50, window_size=250)

pd.DataFrame(X_train).to_csv('data/feature.csv', index=False)
df = pd.read_csv('data/feature.csv')

# main_directory = 'data/'
# output_directory = 'jalla/'

# data = tsfel.dataset_features_extractor(
#                     main_directory, cfg_file, search_criteria='preprocessed_energy_dataset.csv',
#                     time_unit=3600, resample_rate=100, window_size=250,
#                     output_directory=output_directory)

# for column_name in df.columns:
#     # plt.figure(figsize=(12, 10))
#     plt.title(column_name)
#     plt.plot(df[column_name])
#     plt.show()


column_name = 'forecast solar day ahead_Entropy'
plt.title(column_name)
plt.plot(df[column_name])
plt.show()

column_name = 'forecast solar day ahead_Min'
plt.title(column_name)
plt.plot(df[column_name])
# plt.plot(df[column_name])
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# preprocessed_energy_df = preprocessed_energy_df.drop(['time'], axis=1)
# # Perform one-hot encoding
# categorical_columns = preprocessed_energy_df.columns[preprocessed_energy_df.dtypes == object] 
# Find all categorical columns df = pd.get_dummies(df, columns = categorical_columns, drop_first=True)

X = preprocessed_energy_df.values # getting all values as a matrix of dataframe 
sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(X) # standardizing the data@

# pca = PCA()
# X_pca = pca.fit(X_std)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# num_components = 16
# pca = PCA(num_components)  
# X_pca = pca.fit_transform(X_std) # fit and reduce dimension

# pd.DataFrame(pca.components_, columns = preprocessed_energy_df.columns)

pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_std)
print(pca.n_components_)

n_pcs= pca.n_components_

# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = preprocessed_energy_df.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)
print('')
print(set(most_important_names), len(set(most_important_names)))

# Create training / test split
# X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(preprocessed_energy_df[preprocessed_energy_df.columns[preprocessed_energy_df.columns != 'generation solar']], preprocessed_energy_df['generation solar'], test_size=0.25, random_state=1)

# # Standardize the dataset;
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# # Perform PCA
# pca = PCA()
# # Determine transformed features
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)

# plt.plot(X_test_pca[:, 0], X_test_pca[:, 1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')




feat = pd.read_csv('data/selected_features.csv')
trend = pd.read_csv('data/energy_dataset_trends.csv')

# combine and change names of trend columns
for column in trend.columns:
    trend.rename(columns={column: f'trend_{column}'}, inplace=True)
comined = pd.concat([feat, trend], axis=1)

plt.figure(figsize=(12, 10))
cor = comined.corr(numeric_only=True)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

X = comined.drop(['price actual'], axis=1)
y = comined['price actual']
cor_target = abs(cor['price actual'])

#Selecting highly correlated features
relative_fetures = cor_target[cor_target>0.5]
print(relative_fetures) 