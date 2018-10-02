from .Coach import Coach
from .othello.OthelloGame import OthelloGame as Game
from .othello.keras.NNet import NNetWrapper as nn
from .utils import *

args = dotdict({
    'numIters': 300,
    'numEps': 40,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 30,
    'arenaCompare': 11,
    'cpuct': 1,

    'checkpoint': '/home/jduvall/sr/alpha-zero-general/temp2',
    'load_model': False, #True,
    'load_folder_file': ('/home/jduvall/sr/alpha-zero-general/temp2/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 30,

})

def run():
    g = Game(8)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

if __name__=="__main__":
    run()
