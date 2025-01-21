"""Client for the Flower Federated Learning framework."""

import torch
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from glaucoma_fl.task import Net, get_weights, load_data, set_weights, test, train

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs, device=None):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        # print(results)
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        eval_results: dict = test(self.net, self.testloader, self.device)
        return eval_results['loss'], len(self.testloader.dataset), eval_results

def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition = context.node_config["partition"]
    image_folders = {'train': context.run_config["image-folder-train"],
                    'test': context.run_config["image-folder-test"]}
    labels_files = {'train': context.run_config["labels-folder-train"],
                    'test': context.run_config["labels-folder-test"]}
    batch_size = context.run_config["batch-size"]
    device = context.node_config.get("device", None)

    trainloader, valloader = load_data(image_folders, labels_files, partition, batch_size)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, device).to_client()

# Flower ClientApp
app = ClientApp(client_fn)
