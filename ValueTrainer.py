from munch import Munch
import numpy as np
import torch, random
import pickle
from matplotlib import pyplot as plt

import game
from utils import parse_grid
from wrappers import get_best_move,model

# General Parameters
horizon = 2
episodes = 100
config = Munch({
    'n_hidden': 1,
    'hidden_dim': 6,
    'min_val': -5*horizon,
    'max_val': 100*horizon,
    'dropout': 0.25,
    'lr': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'conv_features' : 16,
})

seed = 0

generate_data = True

if __name__ == "__main__":
    device = torch.device("cpu")
    torch.manual_seed(seed)
    if generate_data:
        np.random.seed(seed)
        random.seed(seed)
        rand = random.Random(987)

        # Collect data
        value_buffer = []
        for episode in range(episodes):
            print(f"Episode: {episode}")
            states = [[] for _ in range(4)]
            moves = []
            rewards = []
            gl = game.GameLogic()
            while not gl.is_gameover():
                # parse the boardstate string
                grid = parse_grid(gl.board())
                if gl.moves_left() > horizon:
                    # rotated colors
                    for rot in range(4):
                        # flattening states here, might change it if we decide to use CNN
                        states[rot].append(model.grid2input(grid,rot = rot))

                move = get_best_move(grid,rand)
                _,sdif,_,_ = gl.play(move)
                rewards.append(sdif)
                if gl.moves_left() > horizon-1:
                    moves.append(move)
            values = [sum([rewards[i+j] * 0.6**j for j in range(horizon)]) for i in range(25-horizon)]

            for i in range(4):
                states_torch = torch.FloatTensor(states[i]).to(device)
                moves = torch.FloatTensor(moves).to(device)
                values = torch.FloatTensor(values).to(device)
                value_buffer.append([states_torch,moves,values])

        pickle.dump(value_buffer, open("cnn_buffer_small.pkl", "wb"))

    else:
        value_buffer = pickle.load(open("cnn_buffer.pkl", "rb"))

    vm_onpolicy = model(config=config, device=device)
    vm_onpolicy.train(replay_buffer=value_buffer)

    vm_onpolicy.save("./",prefix="CNN2_")
    print(vm_onpolicy.valid_losses)
    plt.plot(vm_onpolicy.valid_losses, c='r')
    plt.savefig("results/tmp.png")



