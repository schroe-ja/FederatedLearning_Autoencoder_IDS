from flwr.clientapp import ClientApp

from src.tinyfl.sota.client_app import app as pytorch_app
from src.tinyfl.client_app_android import app as android_app

app = ClientApp()

def route(message, context):
    framework = context.node_config.get("framework", "pytorch")

    if framework == "pytorch":
        return pytorch_app(message, context)

    if framework == "android":
        return android_app(message, context)

    raise ValueError(f"Unknown framework: {framework}")

@app.train()
def train(message, context):
    return route(message, context)


@app.evaluate()
def evaluate(message, context):
    return route(message, context)


@app.query()
def query(message, context):
    return route(message, context)