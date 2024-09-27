import time
from sklearn.base import BaseEstimator
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MfccLSTM(nn.Module, BaseEstimator):
    def __init__(self, num_epochs=500, patience=50):
        super(MfccLSTM, self).__init__()
        self.num_epochs = num_epochs
        self.patience = patience

        hidden_size = 32
        input_size = 20
        dropout = 0.2
        num_classes = 36

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.LazyLinear(128)
        self.final_lstm = nn.LSTM(1, 128, batch_first=True, proj_size=64)

        self.fc = nn.LazyLinear(num_classes)

    def forward(self, images, sequences):
        # must return shape (batch_size, num_classes)
        # batch_size: right now is 16
        # num_classes: right now is 36
        x1 = self.conv(images)
        # print(f'input of first lstm: {sequences.shape[1:]}')
        out1, _ = self.lstm(sequences)
        out1_dp = self.dropout(out1)
        # print(f'output of first lstm: {out1_dp.shape[1:]}')
        # print(f'input of second lstm: {out1_dp[:, -1, :].shape[1:]}')
        out2, _ = self.lstm2(out1_dp[:, -1, :])
        out2_dp = self.dropout(out2)
        # print(f'output of second lstm: {out2_dp.shape[1:]}')
        x2 = self.fc2(self.fc1(out2_dp))
        x3 = torch.cat((x1, x2), 1)
        # print(f'output of concatenation: {x3.shape[1:]}')
        # x4 = self.fc3(x3)
        # # print(f'input final lstm: {x4[:,-1,:].shape[1:]}')
        # print(f'x4.shape: {x4.shape[1:]}')
        # x_final = self.final_lstm(x4)
        # # x = self.fc(final_out[:, -1, :])
        x = self.fc(x3)
        return x

    def fit(self, X, y):
        self._optimizer = optim.Adam(self.parameters(), lr=5e-4)
        # same training method but now inside the class
        model = self.to(device)

        # loss criterion
        criterion = nn.CrossEntropyLoss()

        # concatenate so it has the same shape as before
        # self._dataset = np.concatenate((X, y), axis=1)
        self._dataset = [(X[i], y[i]) for i in range(len(X))]
        print(len(self._dataset))
        print(self._dataset[0][0][0].shape)
        print(self._dataset[0][1].shape)
        train_set, val_set = train_test_split(self._dataset, test_size=0.2)
        train_loader = DataLoader(train_set, batch_size=16)
        val_loader = DataLoader(val_set, batch_size=16)

        best_val_acc, epochs_no_imp = 0, 0
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            tic = time.perf_counter()

            # for images, sequences, labels in train_loader:
            for input, labels in train_loader:
                images, sequences = input
                images = images.to(device)
                sequences = sequences.to(device)
                labels = labels.to(device)

                self._optimizer.zero_grad()

                # converting labels to Long to avoid error "not implemented for Int"
                labels = labels.long()

                # Forward pass
                outputs = model(images, sequences)
                loss = criterion(outputs, labels)
                epoch_train_loss += loss.item() * images.size(0)

                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

                # Backward pass
                loss.backward()
                self._optimizer.step()

            toc = time.perf_counter()
            time_taken = toc - tic

            epoch_train_loss /= len(train_loader.dataset)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Evaluation of the model
            model.eval()
            total, correct = 0, 0

            for input, labels in val_loader:
                images, sequences = input
                images = images.to(device)
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs = model(images, sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            #
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Iter Time: {time_taken:.2f}s")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_no_imp = 0
                best_model_state = model.state_dict()  # Save the best model
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= self.patience:
                print(f'Early stopping after {epoch + 1} epochs')
                model.load_state_dict(best_model_state)  # Load the best model
                break
        return self

    def predict(self, X):
        images_test_set = [t[0] for t in X]
        sequences_test_set = [t[1] for t in X]

        images = torch.stack(images_test_set)
        sequences = torch.stack(sequences_test_set)
        images = images.to(device)
        sequences = sequences.to(device)
        model = self.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(images, sequences)
            _, predicted = torch.max(outputs.data, 1)

        pred = []
        keyss = '1234567890QWERTYUIOPASDFGHJKLZXCVBNMÑ+-'
        # phrase = predicted.tolist()
        # for i in range(len(phrase)):
        #     pred.append(keyss[phrase[i]])
        # pred_df = pd.DataFrame(pred)
        # return np.squeeze(pred_df.to_numpy().T)
        return predicted.tolist()