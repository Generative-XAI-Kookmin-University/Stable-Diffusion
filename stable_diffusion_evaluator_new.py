import torch
import numpy as np
from scipy import stats
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm.auto import tqdm
from einops import repeat, rearrange
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json 
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

class EvalDataset(Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model, global_step

class StableDiffusionEvaluator:
    def __init__(self, config_path, model_path, dataset_path, batch_size=10, num_samples=50000, image_size=512, ddim_steps=50, ddim_eta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu', save_generated_images=False, samples_dir='./generated_samples'):
        self.config_path = config_path
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.save_generated_images = save_generated_images
        self.samples_dir = samples_dir
        if self.save_generated_images:
            os.makedirs(self.samples_dir, exist_ok=True)
        self.config = OmegaConf.load(self.config_path)
        self.model, self.global_step = load_model_from_config(self.config, self.model_path)
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.real_features = None
        self.load_real_features()
    
    def get_inception_features(self, images):
        """Extract inception features from images."""
        images = (images + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        self.inception_model.eval()
        with torch.no_grad():
            features = self.inception_model(images)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features
    
    def load_real_features(self):
        """Load and compute features for real images."""
        dataset = EvalDataset(self.dataset_path, self.image_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)
        n_samples_needed = min(self.num_samples, len(dataset))
        n_batches_needed = int(np.ceil(n_samples_needed / self.batch_size))
        features_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Processing real images")):
                if i >= n_batches_needed:
                    break
                batch = batch.to(self.device)
                batch = 2 * batch - 1  # Convert [0, 1] to [-1, 1]
                features = self.get_inception_features(batch)
                features_list.append(features.cpu())
        self.real_features = torch.cat(features_list, dim=0)[:n_samples_needed]
        print(f"Processed {self.real_features.shape[0]} real images")
    
    def calculate_inception_score(self, features, splits=10):
        """Calculate Inception Score from features."""
        scores = []
        subset_size = features.shape[0] // splits
        for k in range(splits):
            subset = features[k * subset_size: (k + 1) * subset_size]
            prob = torch.nn.functional.softmax(subset, dim=1)
            prob = prob.cpu().numpy()
            p_y = np.mean(prob, axis=0)
            kl_d = prob * (np.log(prob + 1e-10) - np.log(p_y + 1e-10))
            kl_d = np.mean(np.sum(kl_d, axis=1))
            scores.append(np.exp(kl_d))
        return np.mean(scores)
    
    def calculate_fid(self, real_features, fake_features):
        """Calculate FID score between real and fake features."""
        mu1 = np.mean(fake_features.cpu().numpy(), axis=0)
        sigma1 = np.cov(fake_features.cpu().numpy(), rowvar=False)
        mu2 = np.mean(real_features.cpu().numpy(), axis=0)
        sigma2 = np.cov(real_features.cpu().numpy(), rowvar=False)
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    def calculate_precision_recall(self, real_features, fake_features, k=3):
        """Calculate precision and recall metrics."""
        real_features = real_features.cpu()
        fake_features = fake_features.cpu()
        real_dists = torch.cdist(real_features, real_features, p=2)
        real_dists.fill_diagonal_(float('inf'))
        kth_dists, _ = real_dists.kthvalue(k, dim=1)
        tau = kth_dists.median().item()
        
        dists_fake2real = torch.cdist(fake_features, real_features, p=2)
        min_dists_fake, _ = dists_fake2real.min(dim=1)
        precision = (min_dists_fake < tau).float().mean().item()
        
        dists_real2fake = torch.cdist(real_features, fake_features, p=2)
        min_dists_real, _ = dists_real2fake.min(dim=1)
        recall = (min_dists_real < tau).float().mean().item()
        
        return precision, recall
    
    def compute_generated_statistics(self):
        """Generate samples and compute statistics."""
        from ldm.models.diffusion.ddim import DDIMSampler
        ddim_sampler = DDIMSampler(self.model)
        fake_features_list = []
        n_rounds = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(n_rounds), desc="Generating samples and computing features"):
                current_batch_size = min(self.batch_size, self.num_samples - i * self.batch_size)
                if current_batch_size <= 0:
                    break
                    
                shape = [current_batch_size, self.model.model.diffusion_model.in_channels, 
                         self.model.model.diffusion_model.image_size, self.model.model.diffusion_model.image_size]
                
                samples, _ = ddim_sampler.sample(
                    S=self.ddim_steps,
                    batch_size=current_batch_size,
                    shape=shape[1:],
                    eta=self.ddim_eta,
                    verbose=False
                )
                
                x_samples = self.model.decode_first_stage(samples)
                
                if self.save_generated_images:
                    for j in range(current_batch_size):
                        if i * self.batch_size + j < 100:  # Save first 100 images
                            sample = x_samples[j]
                            sample = (sample + 1) / 2
                            sample = sample.permute(1, 2, 0).cpu().numpy()
                            sample = (sample * 255).astype(np.uint8)
                            img = Image.fromarray(sample)
                            img.save(os.path.join(self.samples_dir, f'sample_{i*self.batch_size+j:05d}.png'))
                
                features = self.get_inception_features(x_samples)
                fake_features_list.append(features.cpu())
                
                del samples, x_samples, features
                torch.cuda.empty_cache()
                
        fake_features = torch.cat(fake_features_list, dim=0)[:self.num_samples]
        return fake_features
    
    def save_results(self, results):
        """Save evaluation results to file."""
        results_dir = './eval_results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dict = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'global_step': self.global_step,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'ddim_steps': self.ddim_steps,
            'ddim_eta': self.ddim_eta,
            'inception_score': float(results['inception_score']),
            'fid_score': float(results['fid_score']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'timestamp': timestamp
        }
        filename = os.path.join(results_dir, f'eval_results_{self.global_step}_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(output_dict, f, indent=4)
        print(f"\nResults saved to {filename}")
        if self.save_generated_images:
            print(f"Sample images saved to {self.samples_dir}")
    
    def evaluate(self):
        """Evaluate the model and compute metrics."""
        print(f"Starting evaluation for model at global step {self.global_step}")
        print(f"Computing metrics for {self.num_samples} samples...")
        
        # Generate samples and compute features
        fake_features = self.compute_generated_statistics()
        
        # Calculate Inception Score
        is_score = self.calculate_inception_score(fake_features)
        print(f"Inception Score: {is_score:.3f}")
        
        # Calculate FID Score
        fid = self.calculate_fid(self.real_features, fake_features)
        print(f"FID Score: {fid:.3f}")
        
        # Calculate Precision and Recall
        precision, recall = self.calculate_precision_recall(self.real_features, fake_features, k=3)
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        
        results = {
            'inception_score': is_score,
            'fid_score': fid,
            'precision': precision,
            'recall': recall
        }
        
        self.save_results(results)
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Stable Diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the real dataset for FID calculation')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta parameter (0.0 for deterministic)')
    parser.add_argument('--save_images', action='store_true', help='Save generated images')
    parser.add_argument('--samples_dir', type=str, default='./generated_samples', help='Directory to save generated samples')
    args = parser.parse_args()
    
    evaluator = StableDiffusionEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        save_generated_images=args.save_images,
        samples_dir=args.samples_dir
    )
    results = evaluator.evaluate()
    
if __name__ == '__main__':
    main()