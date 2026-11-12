"""if-and-auto: A Flower / PyTorch app."""

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from src.task import Autoencoder,global_evaluate

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    mu: float = context.run_config["mu"]
    input_dim: int = context.run_config["input-dim"]
    #current_round: int = context.run_config["num-server-rounds"]

    #which_dataset: int = context.run_config["which_dataset"]
    
    # Load global model
    global_model = Autoencoder(input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedProx strategy
    strategy = FedProx(fraction_train=fraction_train)

    # Start strategy, run FedProx for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr},{"mu": mu}),
        num_rounds=num_rounds,  
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
