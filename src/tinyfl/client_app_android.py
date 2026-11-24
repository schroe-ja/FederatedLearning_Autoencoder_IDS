import time
import numpy as np
import torch

from urllib.request import Request, urlopen

from flwr.app import (
    Message,
    Context,
    ArrayRecord,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp
from flwr.proto import transport_pb2 as pb


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local Android data."""

    start_time = time.time()

    state = msg.content["arrays"].to_torch_state_dict()

    parameters = pb.Parameters(
        tensors=[
            np.asarray(
                w.detach().cpu().numpy().T,
                dtype="<f4",
            ).tobytes()
            for w in state.values()
        ],
        tensor_type="ND",
    )

    instruction = pb.ServerMessage(
        fit_ins=pb.ServerMessage.FitIns(
            parameters=parameters,
            config={
                "local_epochs": pb.Scalar(
                    sint64=context.run_config["local-epochs"]
                )
            },
        )
    )

    request = Request(
        "http://127.0.0.1:8082",
        data=instruction.SerializeToString(),
        method="POST",
    )

    with urlopen(request, timeout=670) as response:
        result = pb.ClientMessage.FromString(
            response.read()
        )

    if not result.HasField("fit_res"):
        raise RuntimeError("Expected Android FitRes")

    tensors = result.fit_res.parameters.tensors

    if len(tensors) != len(state):
        raise ValueError("Parameter count mismatch")

    updated_state = {}

    for (name, weight), tensor in zip(
        state.items(), tensors
    ):
        array = np.frombuffer(tensor, dtype="<f4")

        array = array.reshape(
            tuple(weight.shape)[::-1]
        ).T.copy()

        if not np.isfinite(array).all():
            raise ValueError("Invalid Android parameters")

        updated_state[name] = torch.from_numpy(array)

    train_loss = float(
        result.fit_res.metrics["train_loss"].double
    )

    val_loss = float(
        result.fit_res.metrics["val_loss"].double
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(updated_state)

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num-examples": result.fit_res.num_examples,
        "avg_training_time": float(
            "{:.2f}".format(training_time)
        ),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local Android data."""

    start_time = time.time()

    state = msg.content["arrays"].to_torch_state_dict()

    parameters = pb.Parameters(
        tensors=[
            np.asarray(
                w.detach().cpu().numpy().T,
                dtype="<f4",
            ).tobytes()
            for w in state.values()
        ],
        tensor_type="ND",
    )

    instruction = pb.ServerMessage(
        evaluate_ins=pb.ServerMessage.EvaluateIns(
            parameters=parameters,
            config={},
        )
    )

    request = Request(
        "http://127.0.0.1:8082",
        data=instruction.SerializeToString(),
        method="POST",
    )

    with urlopen(request, timeout=670) as response:
        result = pb.ClientMessage.FromString(
            response.read()
        )

    if not result.HasField("evaluate_res"):
        raise RuntimeError("Expected Android EvaluateRes")

    val_loss = float(
        result.evaluate_res.metrics["val_loss"].double
    )

    total_examples = int(
        result.evaluate_res.num_examples
    )

    if total_examples == 0:
        raise ValueError("No Android validation samples")

    testing_time = time.time() - start_time

    metrics = {
        "val_loss": float(val_loss),
        "num-examples": int(total_examples),
        "evaluation_time": float(testing_time),
    }

    return Message(
        content=RecordDict({
            "metrics": MetricRecord(metrics)
        }),
        reply_to=msg,
    )