from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def split_data(data):
    """
    Split data into training and testing sets.
    """
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, n_neighbors=5):
    """
    Train a k-NN model to predict diabetes risk.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
