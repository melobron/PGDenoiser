import argparse
from train_L2 import TrainNr2N

# Arguments
parser = argparse.ArgumentParser(description='Train Nr2N')

parser.add_argument('--exp_detail', default='Nr2N training with Estimated Gaussian Noise', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--model', default='DnCNN', type=str)  # DnCNN, UNet
parser.add_argument('--loss', default='L1', type=str)  # L1, L2
parser.add_argument('--n_epochs', default=300, type=int)
parser.add_argument('--offset_epochs', default=0, type=int)
parser.add_argument('--decay_start_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--critic_iter', default=5, type=int)

# Noise parameters
# parser.add_argument('--gauss_std', type=float, default=0.0892)  # 0.0892
parser.add_argument('--poisson_alpha', type=float, default=0.0218)  # 0.0218
parser.add_argument('--gauss_std', type=float, default=0.2176)  # 0.2176

# Transformations
parser.add_argument('--flip_rotate', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.39125)
parser.add_argument('--std', type=float, default=0.23223)

args = parser.parse_args()

# Train Nr2N
train_Nr2N = TrainNr2N(args=args)
train_Nr2N.train()
