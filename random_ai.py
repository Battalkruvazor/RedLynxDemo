import random
from munch import Munch
import graphical, game
from utils import parse_grid
from wrappers import get_best_move,model
import torch

horizon = 2
episodes = 8
config = Munch({
    'n_hidden': 1,
    'hidden_dim': 6,
    'min_val': -5*horizon,
    'max_val': 100*horizon,
    'dropout': 0.25,
    'lr': 0.001,
    'epochs': 400,
    'batch_size': 8,
    'conv_features' : 16,
})


def ai_callback(board, score, moves_left):
    #dir = random.randint(0, 1) == 0
    #return (random.randint(0, 7 if dir else 6), random.randint(0, 8 if dir else 9), dir)
    grid = parse_grid(board)
    move = get_best_move(grid, rand, vf=(vm_onpolicy if moves_left>2 else None))
    return move



def transition_callback(board, move, score_delta, next_board, moves_left):
    pass # This can be used to monitor outcomes of moves

def end_of_game_callback(boards, scores, moves, final_score):
    return True # True = play another, False = Done


if __name__ == '__main__':
    rand = random.Random(1456)
    speedup = 1.0
    device = torch.device("cpu")
    vm_onpolicy = model(config=config, device=device)
    vm_onpolicy.load("./", prefix="CNN_")
    vm_onpolicy.model.eval()
    g = graphical.Game(ai_callback, transition_callback, end_of_game_callback, speedup)
    g.run()
