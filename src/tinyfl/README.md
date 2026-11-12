# TinyFL Model: ML Based IDS with FlowerAI

- [TinyFL Model: ML Based IDS with FlowerAI](#tinyfl-model-ml-based-ids-with-flowerai)
  - [Install dependencies and project](#install-dependencies-and-project)
  - [Run with the Simulation Engine](#run-with-the-simulation-engine)
  - [Run with the Deployment Engine](#run-with-the-deployment-engine)
  - [Customizing the Model](#customizing-the-model)
    - [Use own Dataset](#use-own-dataset)
      - [trainloader](#trainloader)
      - [validaton\_loader](#validaton_loader)
      - [X\_test\_full](#x_test_full)
      - [X\_Validation](#x_validation)
      - [y\_true](#y_true)
      - [X\_train\_classifier](#x_train_classifier)
      - [y\_class](#y_class)
  - [dataset\_load.py](#dataset_loadpy)
    - [For Clients](#for-clients)
    - [For Server Global Model](#for-server-global-model)

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `Python_FL_Model` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Customizing the Model

### Use own Dataset
To use your own Dataset you need to write a custom `dataset_load.py` file. In my example my data was already feature selected, scaled, cleaned of duplicates and inf/NaN values.
The new `dataset_load.py` needs to return these values trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_classifier,y_class.

#### trainloader
This is the pure Benign training data from a dataset saved in a Dataloader.

#### validaton_loader
This is the pure Benign validation data from a dataset saved in a Dataloader.

#### X_test_full
This is a mix of Benign and Anomaly testing data that is used for evaluation. This data is used to calculate the reconstruction error and the Metrics. 
In case of Crossdataset this is from the Testing Dataset.

#### X_Validation
Same data as `validation_loader` only not saved in a Dataloader. Used for calculating unsupervised threshold and normalizing the reconstruction errors for Decision tree `mu_val` and `sigma_val`.
In case of Crossdataset this is from the Training Dataset.
#### y_true
The Ground Truth for X_test_full. 0 for Benign, 1 for Anomaly.

#### X_train_classifier
This is a mix of Benign and Anomaly testing data that is used for fitting the Decision Tree Classifier. 
In case of Crossdataset this is from the Training Dataset.

#### y_class
The Ground Truth for y_class. 0 for Benign, 1 for Anomaly.

## dataset_load.py
change `which_dataset` in `pyproject.yoml` to change Clients training and testing Dataset. change `which_dataset` in `task.py` to change Server Evaluation dataset
### For Clients 
- `load_cross_data` `which_dataset` 0 is CIC-BoTIoT->IoTID20, 1 is IoTID20->CIC-BoTIoT
- `load_mono_dataset` `which_dataset` 0 is CIC-BoTIoT, 1 is IoTID20

### For Server Global Model
- `load_centralized_dataset` `which_dataset` 0 is CIC-BoTIoT, 1 is IoTID20
- `load_crossdataset` `which_dataset` 0 is CIC-BoTIoT->IoTID20, 1 is IoTID20->CIC-BoTIoT

