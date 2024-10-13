import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import pandas as pd

class MfccLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, dropout=0.2):
        super(MfccLSTM, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1,1),
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
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, image_input, sequence_input):
        if sequence_input.dim() == 4:
            sequence_input = sequence_input.squeeze(1)  # Remove the channels dimension if it exists
        if image_input.dim() == 5:
            image_input = image_input.squeeze(1)  # Remove the channels dimension if it exists
        
        x1 = self.conv(image_input)
        #print("Successfully passed conv layer")
        #print(f"X1 shape: {x1.shape}")
        #print(f"Sequence input shape: {sequence_input.shape}")

        #print(f"Sequence input dim: {sequence_input.dim()}")

        # Ensure sequence_input has shape [batch_size, sequence_length, input_size]
        if sequence_input.dim() == 4:
            batch_size, channels, sequence_length, input_size = sequence_input.shape
            sequence_input = sequence_input.view(batch_size, sequence_length, channels * input_size)
        elif sequence_input.dim() == 3:
            sequence_input = sequence_input.permute(0, 2, 1)  # Adjust dimensions if needed

        #print(f"Sequence input shape after view: {sequence_input.shape}")

        # Ensure the input size matches the LSTM's expected input size
        assert sequence_input.size(-1) == self.lstm.input_size, f"Expected input size {self.lstm.input_size}, got {sequence_input.size(-1)}"

        x2, _ = self.lstm(sequence_input)
        x2 = self.dropout(x2)
        x2, _ = self.lstm2(x2)
        x2 = self.fc(x2[:, -1, :])
        
        return x1, x2

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, 1,1)  # Adjusted input channels to 4
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1,1)
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_with_cross_validation(dataset, num_epochs, model_name, num_classes, patience=15, random_state=42, n_splits=10, device='cpu'): #specify device as 'cuda' in run
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}/{n_splits}')
        
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

        model = MfccLSTM(input_size=40000, hidden_size=256, output_size=num_classes, num_classes=num_classes)

        #model = CNN(num_classes=num_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc, epochs_no_imp = 0, 0
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            tic = time.perf_counter()
            
            for images, sequences, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                
                optimizer.zero_grad()

                #print(f"Images shape: {images.shape}")
                # Reshape images to remove the extra dimension
                images = images.squeeze(1)  # Remove the extra dimension

                #print(f"Images shape after squeeze: {images.shape}")

                # Converting labels to Long to avoid error "not implemented for Int"
                labels = labels.long()
                #print(f"num_classes: {num_classes}")
                #print(f"Labels: {labels.unique()}")
                
                # Check that labels are within the valid range
                assert labels.min() >= 0 and labels.max() < num_classes, "Labels are out of bounds"
                
                # Print the shape of the input tensors
                #print(f"Images shape: {images.shape}")
                #print(f"Labels shape: {labels.shape}")

                # Forward pass
                _, outputs = model(images, sequences)
                # Calculate the loss
                loss = criterion(outputs, labels)
            
                #print(f"Outputs shape: {outputs.shape}")
                #print(f"loss shape: {loss.shape}")
                #print(f"images size: {images.size(0)}")

                # Check for NaNs or Infs
                if torch.isnan(loss) or torch.isinf(loss):
                    #print("Loss contains NaNs or Infs")
                    continue

                #print(f"Loss: {loss}")
                epoch_train_loss += loss.item() * images.size(0)

                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
                
                loss.backward()
                optimizer.step()
            
            toc = time.perf_counter()
            time_taken = toc - tic
            
            epoch_train_loss /= len(train_loader.dataset)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            
            #validation phase
            model.eval()
            total, correct = 0, 0
            for images, sequences, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)

                # Reshape images to remove the extra dimension
                images = images.squeeze(1)  # Remove the extra dimension

                _, outputs = model(images, sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Iter Time: {time_taken:.2f}s")
                
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_no_imp = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                model.load_state_dict(best_model_state)
                break
        
        fold_results.append((epoch+1, best_val_acc))
        print(f'Fold {fold+1} Best Validation Accuracy: {best_val_acc:.4f}')
    torch.save(model.state_dict(), model_name)

    return fold_results

def predict_mfcc(dataset, model_path, device_external, keys, num_classes, batch_size=32):
    def get_batches(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    all_preds = []

    device = torch.device(device_external)

    model = MfccLSTM(input_size=40000, hidden_size=256, output_size=num_classes, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for batch in get_batches(dataset, batch_size):
        images_test_set = [t[0] for t in batch]
        sequences_test_set = [t[1] for t in batch]

        images = torch.stack(images_test_set).to(device)
        sequences = torch.stack(sequences_test_set).to(device)

        # Reshape images to remove the extra dimension
        images = images.squeeze(1)  # Remove the extra dimension

        with torch.no_grad():
            _, outputs = model(images, sequences)
            _, predicted = torch.max(outputs.data, 1)

        phrase = predicted.tolist()
        for i in range(len(phrase)):
            all_preds.append(keys[phrase[i]])

    pred_df = pd.DataFrame(all_preds)
    return pred_df
