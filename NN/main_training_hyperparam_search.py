import torch
from VAE import *
from dataloader import *
import os
import matplotlib.pyplot as plt
from plot import plot_param_reconstructions
import random
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
import tempfile
from pathlib import Path
from ray.tune.search.hyperopt import HyperOptSearch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle


def kl_divergence(mean, log_variance, beta):
    # mu.shape = (batch_size, latent)
    kld = torch.sum(-0.5 * (1 + log_variance - mean ** 2 - torch.exp(log_variance)), dim=1)  # (batch_size, 1)
    kld = torch.mean(kld)  # (1)
    return kld * beta


def load_data(data_dir, feature, device, batch_size):
    data = BenjoDataset(data_dir, features=feature_type, device=device)

    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    if True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split*2:], indices[split:split*2], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)
    return train_loader, validation_loader, test_loader


def train_search(config):
    data_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/audio"
    device = "cuda"
    model = VAE(input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                activation=config['activation'], device=device)
    model.to(device)

    criterion = nn.MSELoss(reduction='mean')
    opt = optim.Adam(model.parameters(), lr=config['lr'])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            opt.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loader, valid_loader, test_loader = load_data(data_dir, 'params', DEVICE, config['batch_size'])

    scaler = GradScaler()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config['gamma'])

    for epoch in range(start_epoch, 10):
        training_loss = 0

        for i, x in enumerate(train_loader):
            # x = x.flatten()
            x = x.to(device)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                z, mu, log_var = model.encoder.forward(x)
                x_hat = model.decoder.forward(z)
                recon_loss = criterion(x_hat, x)
                kl = kl_divergence(mu, log_var, config['beta'])
                loss = recon_loss + kl

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            training_loss += loss  # / len(data)

        # scheduler.step()
        validation_loss = 0
        for i, x in enumerate(valid_loader):
            x = x.to(device)
            with torch.no_grad():
                z, mu, log_var = model.encoder.forward(x)
                x_hat = model.decoder.forward(z)
                recon_loss = criterion(x_hat, x)
                kl = kl_divergence(mu, log_var, config['beta'])
                validation_loss += recon_loss + kl

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(),
                       os.path.join(temp_checkpoint_dir, "model.pth"))
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            train.report({"val_loss": validation_loss.item() / len(valid_loader)},
                         checkpoint=checkpoint)


def main(num_samples=10, max_num_epochs=10):
    data_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/audio"
    experiment_name = 'Hyper param search 2'
    storage_path = '/home/midml/Desktop/Leo_project/Benjolin_MA/raytune'

    config = {
        "hidden_dim": 16,
        "input_dim": 8,
        "latent_dim": 2,
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": 32,
        "gamma": 1,
        "beta": tune.loguniform(1e-4, 1e-2),
        "activation": tune.choice(["relu", "tanh", "sigmoid"])
    }

    scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=2,
            reduction_factor=2,
        )

    search_algorithm = HyperOptSearch()

    tuner = tune.Tuner(
        tune.with_resources(
            train_search,
            resources={'cpu': 2, 'gpu': 1}
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            search_alg=search_algorithm,
            scheduler=scheduler,
            metric='val_loss',
            mode='min'
        ),
        run_config=train.RunConfig(stop={"training_iteration": max_num_epochs},
                                   name=experiment_name,
                                   checkpoint_config=train.CheckpointConfig(
                                       checkpoint_score_attribute="val_loss",
                                       num_to_keep=5,
                                   ),
                                   storage_path=storage_path)
    )
    # result = tune.run(...)

    results = tuner.fit()
    result_dataframe = results.get_dataframe()
    dfs = {result.path: result.metrics_dataframe for result in results}

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)

    best_result = results.get_best_result(metric='val_loss', mode='min')
    print("Best result path: ", best_result.path)
    print(best_result.metrics)
    print(best_result.metrics_dataframe)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise Exception("NO GPU AVAILABLE. ABORTING TRAINING")
    DEVICE = "cuda"

    torch.set_default_dtype(torch.float32)
    feature_type = 'params'
    random_seed = 42

    main(num_samples=20, max_num_epochs=10)
