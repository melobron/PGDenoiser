from torch.utils.data import Dataset
from utils import *
import cv2


class Noisy(Dataset):
    def __init__(self, gauss_std, poisson_alpha, train=True, transform=None, noisy_dir='../SEM_data/Hitachi/Single'):
        super(Noisy, self).__init__()

        self.noisy_dir = noisy_dir
        if train:
            self.noisy_dir = os.path.join(self.noisy_dir, 'train')
        else:
            self.noisy_dir = os.path.join(self.noisy_dir, 'test')
        self.noisy_paths = make_dataset(self.noisy_dir)

        self.gauss_std = gauss_std
        self.poisson_alpha = poisson_alpha

        self.transform = transform

    def __getitem__(self, item):
        noisy_path = self.noisy_paths[item]
        noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE) / 255.
        noisy = np.clip(noisy, 0, 1.)
        noisy = np.expand_dims(noisy, axis=2)
        transformed = self.transform(image=noisy)
        return transformed['image']

    def __len__(self):
        return len(self.noisy_paths)

    def add_noise_numpy(self, img):
        gauss_noise = self.gauss_std * np.random.normal(size=img.shape)
        # poisson_noise = np.random.poisson(self.poisson_alpha*img*255.) / (self.poisson_alpha*255.)
        # added = gauss_noise + poisson_noise
        added = img + gauss_noise
        added = np.clip(added, 0, 1.)
        return added

    def add_noise_tensor(self, img):
        gauss_noise = self.gauss_std * torch.randn(size=img.shape).to(img.device)
        # poisson_noise = torch.poisson(self.poisson_alpha*img*255.) / (self.poisson_alpha*255.)
        # added = gauss_noise + poisson_noise
        added = img + gauss_noise
        added = torch.clamp(added, 0, 1.)
        return added
