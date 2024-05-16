import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    algorithms = [
            'KNN',
            'Random Forest Classifier',
            'Decision Tree Classifier',
            'Gaussian Naive Bayes'
    ]

    # prep models
    models = [
            KNeighborsClassifier(n_neighbors=1),
            RandomForestClassifier(n_estimators=100, max_depth=5),
            DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5),
            GaussianNB()
    ]

    # fit models
    results = []
    for model, algo in zip(models, algorithms):
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        results.append({'Algorithm': algo, 'Accuracy': acc})

    model_accs = pd.DataFrame(results)
    model_accs = model_accs.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    print(model_accs)
