import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_wine_data(test_size=0.2, random_state=42):
    red = pd.read_csv('dataset/winequality-red.csv', sep=';')
    white = pd.read_csv('dataset/winequality-white.csv', sep=';')

    red['type'] = 0
    white['type'] = 1

    df = pd.concat([red, white], ignore_index=True)

    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
