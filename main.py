import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

numerical_feature_cols = [
    'Temperature (K)',
    'Luminosity(L/Lo)',
    'Radius(R/Ro)',
    'Absolute magnitude(Mv)'
]

if __name__ == "__main__":
    data = pd.read_csv("data.csv")

    # encode Star Color and Spectral Class from string to int
    label_encoder = LabelEncoder()
    data['Spectral Class'] = label_encoder.fit_transform(data['Spectral Class'])
    data['Star color'] = label_encoder.fit_transform(data['Star color'])


    # scale features
    scaler = StandardScaler()
    numerical_features = data[numerical_feature_cols]
    scaler.fit(numerical_features)
    data[numerical_feature_cols] = scaler.transform(numerical_features)

    # prep data
    X = data.drop(['Star type'], axis=1)
    y = data['Star type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(acc)
