import random
import graphical, game
import copy

def parse_grid(grid_str):
    grid = [[(ord(s) - ord('a') if s != '#' else 4) for s in r] for r in grid_str.split()]
    return grid


def ai_callback(board, score, moves_left):
    #dir = random.randint(0, 1) == 0
    #return (random.randint(0, 7 if dir else 6), random.randint(0, 8 if dir else 9), dir)
    score = -5
    move = (random.randint(0, 7 if dir else 6), random.randint(0, 8 if dir else 9), dir)
    grid = parse_grid(board)
    for dir_ in [0,1]:
        for x in range(8 if dir_ else 7):
            for y in range(9 if dir else 8):
                grid_tmp = copy.deepcopy(grid)
                game.Board._move_in_place(grid_tmp,(x,y,dir_))
                _, scr = game.Board._step_impl(grid_tmp,(x, y, dir_))
                if scr > score:
                    score = scr
                    move = (x, y, dir_)
    return move



def transition_callback(board, move, score_delta, next_board, moves_left):
    pass # This can be used to monitor outcomes of moves

def end_of_game_callback(boards, scores, moves, final_score):
    return True # True = play another, False = Done


if __name__ == '__main__':
    speedup = 1.0
    g = graphical.Game(ai_callback, transition_callback, end_of_game_callback, speedup)
    g.run()
