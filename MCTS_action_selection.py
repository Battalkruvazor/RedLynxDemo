from utils import parse_grid, get_best_move_onesample, play_in_place, matching_moves
import random
import numpy as np
import copy

class _MoveCandidate:
    def __init__(self,move):
        self.move = move
        self.visits = 0
        self.total_score = 0

# a simple MC search implementation
class simple_MCS:
    def __init__(self,nsims,look_ahead, exp_coef, rand):
        self._move_candidates = []
        self._root = None
        self._total_sims = 0
        self._nsims = nsims
        self._exp_coef = exp_coef
        self._look_ahead = look_ahead 
        self._rand = rand
        
    def _initialize(self,board_str):
        self._root = parse_grid(board_str)
        self._total_sims = 0
        self._move_candidates = []
        for i,j,vert in matching_moves(self._root):  
            self._move_candidates.append(_MoveCandidate((i,j,vert)))

    def _select(self,exploration):
        best_score = -100
        best_move = (1,1,1)
        #dir = random.randint(0, 1) == 0 
        #best_move = (random.randint(0, 7 if dir else 6), random.randint(0, 8 if dir else 9), dir)

        for move in self._move_candidates:
            score = move.total_score/move.visits
            if exploration:
                score += self._exp_coef * np.sqrt(1.0/move.visits)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _simulate(self,move):
        grid = copy.deepcopy(self._root)
        scr = play_in_place(grid,move.move,self._rand)
        for _ in range(self._look_ahead):
            mv = get_best_move_onesample(grid,self._rand)
            scr += play_in_place(grid,mv,self._rand)
        move.visits += 1
        move.total_score += scr

    def _evaluate(self):
        for move in self._move_candidates:
            self._simulate(move)
        
        for _ in range(self._nsims):
            move = self._select(exploration = True)
            self._simulate(move)

    def get_move (self,board_str):
        self._initialize(board_str)
        self._evaluate()
        return self._select(exploration=False).move
        
        
