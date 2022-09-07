import torch.optim as optim
from torch.utils.data import DataLoader

from models import Critic, DnCNN, UNet
from dataset import Noisy
from utils import *

import time
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm


class TrainGAN:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.offset_epochs = args.offset_epochs
        self.decay_start_epochs = args.decay_start_epochs
        self.batch_size = args.batch_size
        self.critic_iter = args.critic_iter
        self.lr_g = args.lr_g
        self.lr_c = args.lr_c
        self.weight_clip = args.weight_clip
        self.loss_weight = args.loss_weight

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Noise Parameters
        self.gauss_std = args.gauss_std
        self.poisson_alpha = args.poisson_alpha

        # Models
        if args.model == 'DnCNN':
            self.generator = DnCNN(channels=1).to(self.device)
        elif args.model == 'UNet':
            self.generator = UNet(in_channels=1, out_channels=1).to(self.device)
        self.critic = Critic(channels=1).to(self.device)

        # Model Weight Initialize
        self.critic.apply(weights_init_normal)
        self.generator.apply(weights_init_normal)

        # Loss
        self.criterion_L1 = nn.L1Loss()

        # Optimizer
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_c)

        # Scheduler
        self.g_scheduler = lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=LambdaLR(self.n_epochs, self.offset_epochs, self.decay_start_epochs).step)

        # Transform
        train_transform = A.Compose(get_transforms(args))
        test_transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
            ToTensorV2()
        ])

        # Dataset
        self.train_dataset = Noisy(gauss_std=args.gauss_std, poisson_alpha=args.poisson_alpha, transform=train_transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        self.test_dataset = Noisy(gauss_std=args.gauss_std, poisson_alpha=args.poisson_alpha, transform=test_transform)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)

        # Save Paths
        self.exp_dir, self.exp_num = make_exp_dir('./experiments/')['new_dir'], make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.result_path = os.path.join(self.exp_dir, 'results')
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(args.__dict__, f, indent=4)

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def train(self):
        print(self.device)

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):
            # Training
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for batch, noisy in enumerate(tepoch):
                    noisy = noisy.to(self.device)

                    self.generator.train()
                    self.critic.train()

                    for d_iter in range(self.critic_iter):
                        # Train Discriminator
                        self.c_optimizer.zero_grad()

                        # Real Loss
                        c_loss_real = self.critic(noisy)
                        c_loss_real = c_loss_real.mean()

                        # Fake Loss
                        denoised = self.generator(noisy)
                        renoisy = self.train_dataset.add_noise_tensor(denoised)
                        c_loss_fake = self.critic(renoisy)
                        c_loss_fake = c_loss_fake.mean()

                        # Critic Loss
                        c_loss = -(c_loss_real - c_loss_fake)
                        c_loss.backward()
                        self.c_optimizer.step()

                        # Weight Clipping
                        for p in self.critic.parameters():
                            p.data.clamp_(-self.weight_clip, self.weight_clip)

                    # Train Generator
                    self.g_optimizer.zero_grad()

                    # GAN Loss
                    denoised = self.generator(noisy)
                    renoisy = self.train_dataset.add_noise_tensor(denoised)
                    g_gan_loss = self.critic(renoisy)
                    g_gan_loss = -g_gan_loss.mean()

                    # L1 Loss
                    g_l1_loss = self.criterion_L1(noisy, renoisy)

                    # Generator Loss
                    g_total_loss = g_gan_loss + self.loss_weight*g_l1_loss
                    g_total_loss.backward()
                    self.g_optimizer.step()

                    # Total Loss
                    total_loss = c_loss + g_total_loss

                    tepoch.set_postfix(c_loss=c_loss.item(), g_loss=g_total_loss.item(), loss=total_loss.item())

            # Scheduler
            self.g_scheduler.step(epoch)

            # Save Evaluated Images
            with torch.no_grad():
                sample_image = self.test_dataset[0].to(self.device)
                sample_image = torch.unsqueeze(sample_image, dim=0)

                self.generator.eval()
                sample_denoised = self.generator(sample_image)
                sample_denoised = torch.squeeze(sample_denoised, dim=0).cpu()
                sample_denoised = denorm(sample_denoised, mean=self.mean, std=self.std)
                sample_denoised = torch.clamp(sample_denoised, 0., 1.)

            # Summary Writer
            self.summary.add_scalar('Critic_Loss', c_loss.item(), epoch)
            self.summary.add_scalar('Generator_GAN_Loss', g_gan_loss.item(), epoch)
            self.summary.add_scalar('Generator_L1_Loss', g_l1_loss.item(), epoch)
            self.summary.add_scalar('Total_loss', g_total_loss.item(), epoch)
            self.summary.add_image('Evaluated_image', sample_denoised, epoch, dataformats='CHW')

            # Checkpoints
            if epoch % 100 == 0 or epoch == self.n_epochs:
                torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

        self.summary.close()
