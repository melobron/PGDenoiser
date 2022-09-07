import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DnCNN, UNet
from dataset import Noisy
from utils import *

import time
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm


class TrainNr2N:
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

        # Models
        if args.model == 'DnCNN':
            self.model = DnCNN(channels=1).to(self.device)
        elif args.model == 'UNet':
            self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model.apply(weights_init_normal)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.offset_epochs = args.offset_epochs
        self.decay_start_epochs = args.decay_start_epochs
        self.batch_size = args.batch_size

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Noise Parameters
        self.gauss_std = args.gauss_std
        self.poisson_alpha = args.poisson_alpha

        # Loss
        if args.loss == 'L1':
            self.criterion = nn.L1Loss()
        elif args.loss == 'L2':
            self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.n_epochs, self.offset_epochs, self.decay_start_epochs).step)

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

                    # Training
                    self.model.train()
                    self.optimizer.zero_grad()

                    # Denoise
                    denoised = self.model(noisy)

                    # Add Estimated Noise
                    denoised = denorm(denoised, mean=self.mean, std=self.std)
                    denoised = torch.clamp(denoised, 0, 1.)
                    renoisy = self.train_dataset.add_noise_tensor(denoised)

                    # Loss
                    loss = self.criterion(renoisy, noisy)
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

            # Scheduler
            self.scheduler.step()

            # Save Evaluated Images
            with torch.no_grad():
                sample_image = self.test_dataset[0].to(self.device)
                sample_image = torch.unsqueeze(sample_image, dim=0)

                self.model.eval()
                sample_denoised = self.model(sample_image)
                sample_denoised = torch.squeeze(sample_denoised, dim=0).cpu()
                sample_denoised = denorm(sample_denoised, mean=self.mean, std=self.std)
                sample_denoised = torch.clamp(sample_denoised, 0., 1.)

            # Summary Writer
            self.summary.add_scalar('Train_loss', loss.item(), epoch)
            self.summary.add_image('Evaluated_image', sample_denoised, epoch, dataformats='CHW')

            # Checkpoints
            if epoch % 100 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

        self.summary.close()
