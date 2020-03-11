from munch import Munch
import numpy as np
import torch
import random

import game
from ValueMLP import ValueModel
from utils import parse_grid, get_best_move, one_hot, rotate_cell, grid2input
from MCTS_action_selection import simple_MCS

# General Parameters
horizon = 3
episodes = 500
config = Munch({
    'n_hidden': 1,
    'hidden_dim': 100,
    'min_val': -5*horizon,
    'max_val': 100*horizon,
    'dropout': 0.25,
    'lr': 0.005,
    'epochs': 400,
    'batch_size': 16,
})

seed = 123

if __name__ == "__main__":
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    vm_onpolicy = ValueModel(config=config, device = device)
    vm_onpolicy.load("./",prefix="long_complex_")
    vm_onpolicy.model.eval()

    # Collect validation data
    value_buffer = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        glseed = random.getrandbits(128)

        states = [[] for _ in range(4)]
        moves = []
        rewards= []
        real_values = []
        predicted_values = []
        rand = random.Random(13414)
        gl = game.GameLogic(glseed)
        while not gl.is_gameover():
            # flattening states here, might change it if we decide to use CNN
            grid = parse_grid(gl.board())
            if gl.moves_left() > horizon:
                # rotated colors
                for rot in range(4):
                    states[rot].append(grid2input(grid,rot = rot))
            #print(vm_onpolicy.model.forward(torch.FloatTensor(states[0])).detach().numpy())
            #predicted_values.append(vm_onpolicy.model.forward(torch.FloatTensor(states[0][-1])).detach().numpy())
            move = get_best_move(grid,rand, (vm_onpolicy if gl.moves_left()>3 else None))
            _,sdif,_,_ = gl.play(move)
            rewards.append(sdif)
            if gl.moves_left() > horizon-1:
                moves.append(move)
        values = [sum([rewards[i + j] * 0.8 ** j for j in range(horizon)]) for i in range(25 - horizon)]
        #print(values)
        #print(predicted_values)
        #assert False
        print(f"Value score is {sum(rewards)}")
        


        rand = random.Random(13414)
        states = [[] for _ in range(4)]
        moves = []
        rewards= []
        gl = game.GameLogic(glseed)
        while not gl.is_gameover():
            # flattening states here, might change it if we decide to use CNN
            grid = parse_grid(gl.board())
            if gl.moves_left() > horizon:
                # rotated colors
                for rot in range(4):
                    states[rot].append(grid2input(grid,rot = rot))

            move = get_best_move(grid,rand)
            _,sdif,_,_ = gl.play(move)
            #print(f"Greedy, selected move: {move}, with reward: {sdif}")
            rewards.append(sdif)
            if gl.moves_left() > horizon-1:
                moves.append(move)
        print(f"Greedy score is {sum(rewards)}")

        rand = random.Random(13414)
        states = [[] for _ in range(4)]
        moves = []
        rewards= []
        mcs = simple_MCS(nsims = 100, look_ahead = 1, exp_coef = 190, rand = rand)
        gl = game.GameLogic(glseed)
        while not gl.is_gameover():
            #print("step")

            move = mcs.get_move(gl.board())
            _,sdif,_,_ = gl.play(move)
            #print(f"real reward: {sdif}")
            rewards.append(sdif)
        print(f"MCS score is {sum(rewards)}")