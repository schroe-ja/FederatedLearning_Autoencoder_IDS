import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyfl.evaluation_metrics import EvaluationMetrics
from src.tinyfl import constants

class Autoencoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4) -> None: 
        super(Autoencoder, self).__init__()
        # Encoder: Compresses data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64), # Helps training stability
            nn.ReLU(),      
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 12)         
        )
        
        # Decoder: Reconstructs data
        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, input_dim) 
        )
    
    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

def phase_recall(y_pred_malicious, phase_malicious_test, model_name):
    
    y_pred = np.asarray(y_pred_malicious).ravel()
    phases = np.asarray(phase_malicious_test).ravel()

    if len(y_pred) != len(phases):
        raise ValueError("Predictions and phase labels have different lengths")

    results = {}

    for phase in ["scan", "distribution"]:
        mask = phases == phase

        if not np.any(mask):
            print(f"{model_name} - {phase}: no test samples")
            continue

        recall = float(np.mean(y_pred[mask] == 1))
        results[phase] = recall

        print(
            f"{model_name} - {phase} detection rate: "
            f"{recall:.2%} "
            f"({np.sum(y_pred[mask] == 1)}/{np.sum(mask)})"
        )

    return results

def malicious_probability(model, X):
    probabilities = model.predict_proba(X)

    class_indices = np.where(model.classes_ == 1)[0]

    if len(class_indices) == 0:
        raise ValueError("Decision Tree was not trained with malicious samples.")

    return probabilities[:, class_indices[0]]

# weighted reconstruction loss
def _reconstruction_loss(outputs, inputs):
    idx = {
        name: i for i, name in enumerate(constants.FEATURES)
    }

    feature_weights = torch.ones(
        inputs.shape[1],
        device=inputs.device
    )

    # Give more importance to event-specific information
    important_features = [
        "url",
        "command_hash",
        "username_hash",
        "protocol_hash",
        "successful",
    ]

    for feature in important_features:
        feature_weights[idx[feature]] = 2.0

    for c in constants.BINFO_NUMERIC_COLS:
        feature_weights[idx[f"binfo_{c}"]] = 2.0

    for c in constants.BINFO_TEXT_COLS:
        feature_weights[idx[f"binfo_{c}"]] = 2.0

    loss = (outputs - inputs) ** 2

    return (loss * feature_weights).mean()

def train(net, trainloader, validaton_loader, partition_id, epochs, lr, mu, device):
    """Train the model on the training set."""

    EvaluationMetrics.print_memory_usage()

    net.to(device)

    global_params = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05) # weight_decay L2 regularization
    criterion = _reconstruction_loss

    print("training device", partition_id)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        for data in trainloader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #loss = criterion(outputs, inputs)
            
            #FedProx           
            proximal_term = 0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                 proximal_term += torch.sum((local_weights - global_weights.to(device))**2)
            loss = criterion(outputs, inputs) + (mu / 2.0) * proximal_term
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/len(trainloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for data in validaton_loader:
                inputs = data[0].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validaton_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            #best_model_wts = copy.deepcopy(net.state_dict()) 
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop if overfitting detected
        if patience_counter >= patience:
            print(f'\n Early stopping triggered at epoch {epoch+1}')
            print(f'Validation loss has not improved for {patience} epochs')
            print(f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            break

    EvaluationMetrics.print_memory_usage()  
     
    avg_trainloss = train_loss/len(trainloader)
    avg_valloss = val_loss / len(validaton_loader)
    print(f'Train Loss: {avg_trainloss:.6f} | Val Loss: {avg_valloss:.6f}')

    return avg_trainloss, avg_valloss