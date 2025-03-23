import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_and_clean_data

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        # it will use the already created function load_and_clean_data in preprocess.py file
        self.df = load_and_clean_data(self.df)
        return self.df

class Trainer:
    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.save_path)
        print(f"Model saved to {self.save_path}")

class Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

# Example usage
if __name__ == "__main__":
    data_loader = DataLoader("data/raw/sample.csv")
    df = data_loader.load_data()

    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.preprocess()

    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    trainer = Trainer(model, "model/model.pkl")
    trainer.train(X_train, y_train)

    evaluator = Evaluator("model/model.pkl")
    accuracy = evaluator.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
