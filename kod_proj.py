import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.ensemble import IsolationForest

# Wczytanie danych
parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets.values.ravel()

print("Zbiór y (195 wartości):")
print(y[:195])  # Wypisanie 195 wartości ze zbioru y

# Usunięcie zduplikowanych kolumn
X = X.loc[:, ~X.columns.duplicated()]

# Wyświetlenie kształtu danych
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Wyświetlenie kilku wierszy danych
print(X.head())
print(y[:5])

# Sprawdzenie liczby unikalnych osób
#if 'name' in X.columns:
#   unique_people = X['name'].nunique()
#  print(f"Liczba unikalnych osób: {unique_people}")

# Sprawdzenie liczby nagrań dla różnych osób
num_parkinsons_recordings = (y == 1).sum() #Chorzy
print(f"Liczba nagrań od osób z chorobą Parkinsona: {num_parkinsons_recordings}") 

num_healthy_recordings = (y == 0).sum() #Zdrowi
print(f"Liczba nagrań od osób zdrowych: {num_healthy_recordings}")

# Jeśli kolumna 'name' istnieje w danych, usuń ją przed przetwarzaniem
if 'name' in X.columns:
    X_numeric = X.drop(columns=['name'])
else:
    X_numeric = X

# Wybranie tylko kolumn numerycznych
numeric_columns = X_numeric.select_dtypes(include=np.number).columns.tolist()

# Wyświetlenie metadanych i zmiennych
print(parkinsons.metadata)
print(parkinsons.variables)

# Podstawowe statystyki opisowe
print(X.describe())

# Wyświetlenie nazw kolumn
print("Nazwy kolumn:", X.columns.tolist())

# Zapisanie danych do Excel'a
X_numeric.to_excel('parkinsons_data.xlsx', index=False)

# Wizualizacja danych z użyciem boxplotów
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=X_numeric[col])
    plt.title(f'Boxplot dla kolumny: {col}')
    plt.show()

# Wykrywanie i usuwanie outlierów z użyciem z-score
z_scores = np.abs(zscore(X_numeric))
outliers = (z_scores > 3.0)

X_no_outliers = X_numeric[~np.any(outliers, axis=1)]

print(f"Liczba wierszy po usunięciu outlierów: {X_no_outliers.shape[0]}")

# Wykrywanie outlierów za pomocą Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
outliers_if = clf.fit_predict(X_numeric)

# Dodanie kolumny z outlierami do X_numeric
X_numeric['outlier'] = np.where(outliers_if == -1, 1, 0)

print("Liczba outlierów:", X_numeric['outlier'].sum())

# Konwersja y na pandas.Series
y_series = pd.Series(y)

# Wyświetlenie liczby próbek przed zrównoważeniem klas
print("\nLiczba próbek przed zrównoważeniem klas:")
print(y_series.value_counts())

# Standaryzacja danych bez outlierów
scaler = StandardScaler()
scaled_features_no_outliers = scaler.fit_transform(X_no_outliers)

# Oversampling danych (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(scaled_features_no_outliers, y_series[X_no_outliers.index])

# Stworzenie nowego DataFrame ze zrównoważonymi danymi
X_balanced = pd.DataFrame(X_resampled, columns=X_no_outliers.columns)
X_balanced['target'] = y_resampled

# Sprawdzenie nowej liczby próbek
print("\nLiczba próbek po zrównoważeniu klas:")
print(X_balanced['target'].value_counts())

# Trenowanie modelu i ocena jakości
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced.drop(columns=['target']), X_balanced['target'], test_size=0.3, random_state=42)

model_balanced = SVC()
model_balanced.fit(X_train_balanced, y_train_balanced)
predictions_balanced = model_balanced.predict(X_test_balanced)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_balanced, predictions_balanced))
print("\nClassification Report:")
print(classification_report(y_test_balanced, predictions_balanced))

# Optymalizacja parametrów za pomocą Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
grid_balanced = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_balanced.fit(X_train_balanced, y_train_balanced)

print("\nNajlepsze parametry znalezione przez Grid Search:")
print(grid_balanced.best_params_)

grid_predictions_balanced = grid_balanced.predict(X_test_balanced)

print("\nConfusion Matrix po optymalizacji:")
print(confusion_matrix(y_test_balanced, grid_predictions_balanced))
print("\nClassification Report po optymalizacji:")
print(classification_report(y_test_balanced, grid_predictions_balanced))

# Wizualizacja danych po usunięciu outlierów
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=X_no_outliers[col])
    plt.title(f'Boxplot dla kolumny: {col} (bez outlierów)')
    plt.show()
