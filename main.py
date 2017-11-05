import argparse
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', help='train flag')
    parser.add_argument('-data', type=str, default='small', help='which dataset, small or big')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size (default 50)')
    parser.add_argument('-patience', type=int, default=15, help='early stopping patience (default 10)')
    parser.add_argument('-dev', type=str, default=-1, help='what cpu or gpu (recommended) use to train the model')
    args = parser.parse_args()

    if args.train:
        train(args)
