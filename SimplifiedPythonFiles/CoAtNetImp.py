# from new version: tread lightly, adjusted to fit current settings
import sys
sys.path.insert(1,'../ClassificationCNN/')

import time
from coatnet import CoAtNet as CoAtNetImp # Import the specific class
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

# CoAtNet parameters
num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D

def train_coatnet_with_cross_val(dataset, num_epochs, model_name, device_external, num_classes, patience=10): #not using folds?
    # Split dataset into training and validation sets
    train_set, val_set = train_test_split(dataset, test_size=0.2) # using train_test_split instead of subset this time around
    train_loader, val_loader = DataLoader(train_set, batch_size=16), DataLoader(val_set, batch_size=16)
    
    # Initialize model, optimizer, and loss function
    model = CoAtNetImp((64, 64), 1, num_blocks, channels, num_classes=num_classes) #using coatnet instead of mfcclstm
    device = torch.device(device_external) #default to mps
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    
    best_val_acc, epochs_no_imp = 0, 0
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        tic = time.perf_counter()
        
        for images, labels in train_loader: #not using sequences
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
    
            labels = labels.long() # converting labels to Long to avoid error "not implemented for Int"

            # Check that labels are within the valid range
            assert labels.min() >= 0 and labels.max() < num_classes, "Labels are out of bounds"
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_train_loss += loss.item() * images.size(0)
    
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Iter Time: {time_taken:.2f}s")
            
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
        if epochs_no_imp >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    torch.save(model.state_dict(), model_name)
    return epoch+1, best_val_acc

def predict(dataset, model_obj, argnames, model_path, device_external, keys, batch_size=32):
    def get_batches(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    all_preds = []

    # specify device: default to mps
    device = torch.device(device_external)

    # model specifying
    model = model_obj.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for batch in get_batches(dataset, batch_size):
        fin_dict = {}
        
        # create the list with each of the ith range tuples
        for i in range(len(batch[0])-1):
            fin_dict[argnames[i]] = [t[i] for t in batch]
        
        # torch.stack each one of the lists
        for key in fin_dict.keys():
            fin_dict[key] = torch.stack(fin_dict[key]).to(device)
        
        with torch.no_grad():
            outputs = model(**fin_dict)
            _, predicted = torch.max(outputs.data, 1)
        
        phrase = predicted.tolist()
        for i in range(len(phrase)):
            all_preds.append(keys[phrase[i]])

    pred_df = pd.DataFrame(all_preds)
    return pred_df