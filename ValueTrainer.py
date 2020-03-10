from munch import Munch
import numpy as np
import torch, random

import game
from ValueMLP import ValueModel
from utils import parse_grid, get_best_move, one_hot, rotate_cell,grid2input

# General Parameters
horizon = 3
episodes = 500
config = Munch({
    'n_hidden': 2,
    'hidden_dim': 50,
    'min_val': -5*horizon,
    'max_val': 100*horizon,
    'dropout': 0.25,
    'lr': 0.005,
    'epochs': 400,
    'batch_size': 16,
})

seed = 0

if __name__ == "__main__":
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    vm_onpolicy = ValueModel(config=config, device = device)
    
    # Collect data
    value_buffer = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        states = [[] for _ in range(4)]
        moves = []
        rewards= []
        gl = game.GameLogic()
        while not gl.is_gameover():
            # parse the boardstate string
            grid = parse_grid(gl.board())
            if gl.moves_left() > horizon:
                # rotated colors
                for rot in range(4):
                    # flattening states here, might change it if we decide to use CNN
                    states[rot].append(grid2input(grid,rot = rot))

            move = get_best_move(grid)
            _,sdif,_,_ = gl.play(move)
            rewards.append(sdif)
            if gl.moves_left() > horizon-1
                moves.append(move)
        values = [sum([rewards[i+j] * 0.8**j for j in range(horizon)]) for i in range(25-horizon)]

        for i in range(4):
            states_torch = torch.FloatTensor(states[i]).to(device)
            moves = torch.FloatTensor(moves).to(device)
            values = torch.FloatTensor(values).to(device)
            value_buffer.append([states_torch,moves,values])

        
    vm_onpolicy.train(replay_buffer=value_buffer)

    vm_onpolicy.save("./",prefix="long_complex_")


