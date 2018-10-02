from .othello.tourney_files.my_core import MyCore # for compatability, will use other one irl
from .othello.OthelloGame import OthelloGame as Game
from .othello.keras.NNet import NNetWrapper as nn
from .MCTS import MCTS
from .main import args
import numpy as np
from sys import argv, exit, stderr
import subprocess as sb
import time

def convert_player(old_player):
    if old_player == 'o':
        return 1
    elif old_player == '@':
        return -1
    else:
        return 0

class Strategy(MyCore):
    def __init__(self):
        self.act_init(2)

    def act_init(self, num):
        self.args = args
        self.args.numMCTSSims = num
        self.game = Game(8)
        self.nnet = nn(self.game)
        self.nnet.load_checkpoint("temp", "best.pth.tar")
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def convert_board(self, old_board):
        board = [None]*self.game.n
        for y in range(self.game.n):
            row = (y+1)*(self.game.n+2)
            board[y] = [convert_player(old_board[row+(x+1)])
                        for x in range(self.game.n)]
        return np.array(board)

    def best_strategy(self, board, player, move, flag, timelimit=5):
        time_begin = time.time()
        actBoard = self.convert_board(board)
        canonicalBoard = self.game.getCanonicalForm(actBoard, convert_player(player))
        going = True
        while going: 
            pi = self.mcts.getActionProb(canonicalBoard, temp=0)
            action = np.random.choice(len(pi), p=pi)
            action_out = (action//self.game.n+1)*(self.game.n+2)+action%self.game.n+1
            #stderr.write(str((action_out))+"\n")
            move.value = action_out
            going = time.time()-time_begin < timelimit
