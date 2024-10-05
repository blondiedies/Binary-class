from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from coatnet import CoAtNet as CoAtNetImp
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import time
from torch.utils.data import DataLoader


num_blocks = [2, 2, 3, 5, 2]  # L
channels = [64, 96, 192, 384, 768]  # D
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CoAtNet(nn.Module, BaseEstimator):
    def __init__(self, num_epochs=500, patience=15):
        super(CoAtNet, self).__init__()
        self.keys = '1234567890QWERTYUIOPASDFGHJKLZXCVBNMÑ+-'
        self.model = CoAtNetImp((64, 64), 1, num_blocks, channels, num_classes=len(self.keys))
        self.num_epochs = num_epochs
        self.patience = patience

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y):
        # concatenate so it has the same shape as before
        self._dataset = [(X[i], y[i]) for i in range(np.array(X).shape[0])]
        # dataset = np.concatenate((X, y), axis=1)
        train_set, val_set = train_test_split(self._dataset, test_size=0.2)
        train_loader, val_loader = DataLoader(train_set, batch_size=16), DataLoader(val_set, batch_size=16)

        # Initialize model, optimizer, and loss function
        self._optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

        # same training method but now inside the class
        model = self.model.to(device)

        # loss criterion
        criterion = nn.CrossEntropyLoss()

        best_val_acc, epochs_no_imp = 0, 0
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            tic = time.perf_counter()

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                self._optimizer.zero_grad()

                # converting labels to Long to avoid error "not implemented for Int"
                labels = labels.long()

                # Forward pass
                outputs = model(images)
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

            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 1 == 0:
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
        argnames = ["x"]
        fin_dict = {}
        # create the list with each of the ith range tuples
        for i in range(len(X[0]) - 1):
            fin_dict[argnames[i]] = [t[i] for t in self._dataset]

        # torch.stack each one of the lists
        for key in fin_dict.keys():
            fin_dict[key] = torch.stack(fin_dict[key]).to(device)

        X = torch.tensor(np.array(X)).to(device)

        # model specifying
        model = self.model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

        pred = []
        # phrase = predicted.tolist()
        # for i in range(len(phrase)):
        #     pred.append(self.keys[phrase[i]])
        #
        # pred_df = pd.DataFrame(pred)
        # return np.squeeze(pred_df.to_numpy().T)
        return predicted.tolist()