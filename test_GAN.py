import argparse
from models import DnCNN, UNet
import cv2
from utils import *
from pathlib import Path

# Arguments
parser = argparse.ArgumentParser(description='Test Nr2N')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=5, type=int)

# Training parameters
parser.add_argument('--model', default='DnCNN', type=str)  # DnCNN, UNet
parser.add_argument('--n_epochs', default=300, type=int)
parser.add_argument('--aver_num', default=10, type=int)

# Noise parameters
parser.add_argument('--gauss_std', type=float, default=0.0892)  # 0.0892
parser.add_argument('--poisson_alpha', type=float, default=0.0218)  # 0.0218

# Transformations
parser.add_argument('--flip_rotate', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.39125)
parser.add_argument('--std', type=float, default=0.23223)

opt = parser.parse_args()


def denoise_with_patches(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Model
    model = DnCNN(channels=1).to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Transform
    transform = A.Compose(get_transforms(args))

    # Dataset
    data_dir = os.path.join(Path(__file__).parents[1], 'SEM_data')
    noisy_dir = os.path.join(data_dir, 'Hitachi/Single/test')
    noisy_paths = make_dataset(noisy_dir)

    # Save Directory
    save_dir = './experiments/exp{}/results/'.format(args.exp_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Evaluation
    for index, noisy_path in enumerate(noisy_paths):
        noisy_numpy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE) / 255.
        noisy_numpy = np.clip(noisy_numpy, 0, 1.)

        noisy = np.expand_dims(noisy_numpy, axis=2)
        noisy = transform(image=noisy)['image']
        noisy = torch.unsqueeze(noisy, dim=0)
        noisy = noisy.to(device)

        # Single Predictions
        with torch.no_grad():
            noisy_prediction = model(noisy)

        # Transform
        noisy_prediction = torch.squeeze(noisy_prediction).cpu()
        noisy_prediction = denorm(noisy_prediction, mean=args.mean, std=args.std)
        noisy_prediction = torch.clamp(noisy_prediction, 0, 1.).unsqueeze(dim=2).numpy()

        noisy_numpy = np.clip(noisy_numpy * 255., 0, 255)
        noisy_prediction = np.clip(noisy_prediction * 255., 0, 255)

        # Visualize prediction
        save_num = os.path.basename(noisy_path).split('.')[0]
        save_image_dir = os.path.join(save_dir, '{}th_image'.format(save_num))
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        for name, img in {"noisy": noisy_numpy, "prediction": noisy_prediction}.items():
            title = '{}.png'.format(name)
            save_image_path = os.path.join(save_image_dir, title)
            cv2.imwrite(save_image_path, img)

        print('({}/{})th image evaluated'.format(index + 1, len(noisy_paths)))

if __name__ == "__main__":
    denoise_with_patches(opt)
