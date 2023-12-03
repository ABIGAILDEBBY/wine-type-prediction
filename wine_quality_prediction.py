import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from keras.models import Sequential
from keras.layers import Dense


class WineQualityPrediction:
    def __init__(self, red_url, white_url):
        self.red_data = pd.read_csv(red_url, sep=";")
        self.white_data = pd.read_csv(white_url, sep=";")
        self.prepare_data()

    def prepare_data(self):
        self.red_data["type"] = 1
        self.white_data["type"] = 0
        self.wines = self.red_data._append(self.white_data, ignore_index=True)
        self.X = self.wines.iloc[:, :11]
        self.y = np.ravel(self.wines.type)

    def display_dataset_info(self):
        print(f"Dataset Shape: {self.wines.shape}")
        print("\nDataset Description:")
        print(self.wines.describe())

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.34, random_state=45
        )
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation="relu", input_shape=(11,)))
        model.add(Dense(9, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def train_model(self, model, X_train, y_train):
        history = model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
        return history

    def evaluate_model(self, model, X_test, y_test, pdf_pages):
        y_pred = model.predict(X_test)
        y_pred_rounded = y_pred.round()

        accuracy = accuracy_score(y_test, y_pred_rounded)
        precision = precision_score(y_test, y_pred_rounded)
        recall = recall_score(y_test, y_pred_rounded)
        f1 = f1_score(y_test, y_pred_rounded)

        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_rounded)
        self.plot_confusion_matrix(cm, pdf_pages, "Confusion Matrix")

        # Training History
        history = self.train_model(model, X_test, y_test)
        self.plot_training_history(history, pdf_pages, "Training History")

    def plot_confusion_matrix(self, cm, pdf_pages, title):
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        pdf_pages.savefig()
        plt.close()

    def plot_training_history(self, history, pdf_pages, title):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["accuracy"], label="Accuracy")
        plt.plot(history.history["loss"], label="Loss")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()
        pdf_pages.savefig()
        plt.close()

    def run(self):
        pdf_pages = PdfPages("wine_quality_analysis.pdf")

        self.display_dataset_info()
        X_train, X_test, y_train, y_test = self.split_data()
        model = self.build_model()
        self.evaluate_model(model, X_test, y_test, pdf_pages)

        pdf_pages.close()


if __name__ == "__main__":
    red_wine_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    wine_predictor = WineQualityPrediction(red_wine_url, white_wine_url)
    wine_predictor.run()
