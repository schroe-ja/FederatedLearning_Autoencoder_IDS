import torch
import torch.nn as nn
#from torchviz import make_dot
class Autoencoder(nn.Module):
    def __init__(self, input_dim,dropout_rate=0.4): #input dimension hardcoded
        super(Autoencoder, self).__init__()
# Encoder: Compresses data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),      # Helps training stability
            nn.ReLU(),      
            
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(24, 6)         
        )
        
        # Decoder: Reconstructs data
        self.decoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(0.2),
            
            nn.Linear(48, input_dim) 

        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
# Modell bauen und laden
device = torch.device("cpu") # Always export from CPU
model = Autoencoder(67)

model.load_state_dict(torch.load("final_model.pt",map_location=device))
model.eval()

# To get just the encoder
model_enc = model.encoder
model_enc.eval()

# 2. Create a dummy input with the exact shape the ESP32 will use (Batch=1, Features=67)
dummy_input = torch.randn(1, 67, dtype=torch.float32)

# 3. Export Autoencoder to ONNX
print("Exporting Autoencoder to ONNX...")
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    export_params=True,
    opset_version=13, # Opsets 13 or 14 are best for TFLite
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# 4. Export Encoder to ONNX
print("Exporting Encoder to ONNX...")
torch.onnx.export(
    model_enc, 
    dummy_input, 
    "model_encoder.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("ONNX Export Complete!")
print("ONNX model saved: model.onnx")
print("ONNX model saved: model_encoder.onnx")