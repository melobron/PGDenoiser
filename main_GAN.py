import argparse
from train_GAN import TrainGAN

# Arguments
parser = argparse.ArgumentParser(description='Train GAN')

parser.add_argument('--exp_detail', default='GAN training with Estimated Gaussian Noise', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--model', default='DnCNN', type=str)  # DnCNN, UNet
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--offset_epochs', default=0, type=int)
parser.add_argument('--decay_start_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr_g', default=4e-4, type=float)
parser.add_argument('--lr_c', default=1e-6, type=float)
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--weight_clip', default=0.02, type=int)
parser.add_argument('--loss_weight', default=0.0, type=int)

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
train_GAN = TrainGAN(args=args)
train_GAN.train()
