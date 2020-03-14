from munch import Munch
import numpy as np
import torch, random
import pickle
from matplotlib import pyplot as plt
import os

import game
from utils import parse_grid
from wrappers import get_best_move,model

# General Parameters
horizon = 2
episodes = 1000
config = Munch({
    'n_hidden': 1,
    'hidden_dim': 6,
    'min_val': -5*horizon,
    'max_val': 100*horizon,
    'dropout': 0.25,
    'lr': 0.001,
    'epochs': 200,
    'batch_size': 16,
    'conv_features' : 16,
})

seed = 0

generate_data = False

if __name__ == "__main__":
    losses = []
    for sample in range(400):
        print(sample)
        config.hidden_dim = random.randint(3,9)
        config.dropout = 0.5 * random.random()
        config.lr = np.power(0.1,random.randint(1,6))
        config.conv_features = random.randint(8,64)
        dir_ = f"results/result{sample}"
        try:
            os.mkdir(dir_)
        except:
            pass
        device = torch.device("cpu")
        torch.manual_seed(seed)
        value_buffer = pickle.load(open("cnn_buffer_small.pkl", "rb"))

        vm_onpolicy = model(config=config, device=device)
        vm_onpolicy.train(replay_buffer=value_buffer)

        vm_onpolicy.save(dir_,prefix="CNN__")
        plt.plot(vm_onpolicy.valid_losses, c='r')
        plt.savefig(f"{dir_}/validation_curve.png")
        plt.figure()

        losses.append((sample,min(vm_onpolicy.valid_losses)))

        with open(f"{dir_}/parameters.txt","w+") as f:
            f.write(str(config))

        with open(f"{dir_}/results.txt","w+") as f:
            f.write(str(vm_onpolicy.valid_losses))

    losses.sort(key=lambda r: r[1])
    print(losses[:5])

