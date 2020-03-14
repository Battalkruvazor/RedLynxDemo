import copy, random, torch
import game


from ValueMLP import ValueModel as ValueMLP
from ValueCNN import ValueModel as ValueCNN


models = {
    "MLP": ValueMLP,
    "CNN": ValueCNN,
}

model = models["CNN"]

def matches_anything(grid, move, rand):
    gridprime = copy.deepcopy(grid)
    if game.Board._move_in_place(gridprime, move):
        _, sdif = game.Board._step_impl(gridprime, rand)
        return sdif > 0
    else:
        return False


def matching_moves(grid, rand):
    ret = []
    for x in range(8):
        for y in range(10):
            for d in [False, True]:
                mv = (x, y, d)
                if matches_anything(grid, mv, rand):
                    ret.append(mv)
    return ret


def play_in_place(grid, mv, rand):
    game.Board._move_in_place(grid, mv)
    cont = True
    scr = 0
    while cont:
        cont, scrt = game.Board._step_impl(grid, rand)
        scr += scrt
    return scr


def get_best_move(grid, rand, num_sample = 10, vf=None):
    score = -5
    d = random.randint(0, 1)
    move = (random.randint(0, 7 if d else 6), random.randint(0, 8 if d else 9), d)
    moves = matching_moves(grid, rand)
    for mv in moves:
        scr = 0
        for _ in range(num_sample):
            grid_tmp_s = copy.deepcopy(grid)
            scr += play_in_place(grid_tmp_s, mv, rand)
            if vf:
                scr += 0.6 * vf.model.forward(torch.FloatTensor([model.grid2input(grid_tmp_s)]))
        # scr/=3.0
        if scr > score:
            score = scr
            move = mv

    # print(scr)
    return move
