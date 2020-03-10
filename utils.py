import copy, random, torch
import game
from ValueMLP import ValueModel

def parse_grid(grid_str):
    grid = [[(ord(s) - ord('a') if s != '#' else 4) for s in r] for r in grid_str.split()]
    return grid

def one_hot(cell):
    return [1 if cell == i else 0 for i in range(5)]

def rotate_cell(cell, rot = 1):
    if cell<4:
        return (cell+rot)%4
    else: 
        return cell

def matching_moves(grid):
    ret = []
    for x in range(8):
        for y in range(10):
            for d in [False, True]:
                mv = (x, y, d)
                if game.Board._matches_anything(grid, mv):
                    ret.append(mv)
    return ret

def grid2input(grid,rot = 0):
    return [c for r in grid for s in r for c in one_hot(rotate_cell(s,rot=rot))]

def play_in_place(grid,mv):
    game.Board._move_in_place(grid, mv)
    cont = True
    scr = 0
    while cont:
        cont, scrt = game.Board._step_impl(grid, random.Random())
        scr += scrt
    return scr

def get_best_move(grid, vf=None):
    score = -5
    d = random.randint(0,1) 
    move = (random.randint(0, 7 if d else 6), random.randint(0, 8 if d else 9), d)
    move2 = ()
    moves = matching_moves(grid)
    for mv in moves:
        scr = 0
        for _ in range(10):
            grid_tmp_s = copy.deepcopy(grid)
            scr += play_in_place(grid_tmp_s,mv)
            if vf:
                scr += 0.4*vf.model.forward(torch.FloatTensor(grid2input(grid_tmp_s)))
        #scr/=3.0
        if scr > score:
            score = scr
            move = mv

    #print(scr)
    return move