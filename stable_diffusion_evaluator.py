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

# Add imports for Stable Diffusion
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
        img = img.convert('RGB')  # Ensure RGB format
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
    def __init__(
        self,
        config_path,
        model_path,
        dataset_path,
        batch_size=10,
        num_samples=50000,
        image_size=512,
        ddim_steps=50,
        ddim_eta=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        
        # Load configuration
        self.config = OmegaConf.load(self.config_path)
        
        # Load model
        self.model, self.global_step = load_model_from_config(self.config, self.model_path)
        
        # Initialize inception model for FID calculation
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        
        # Load reference dataset features
        self.load_real_features()
    
    def load_real_features(self):
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
                # Scale images to [-1, 1] as per stable diffusion requirements
                batch = 2 * batch - 1
                features = self.get_inception_features(batch)
                features_list.append(features)
                
        self.real_features = torch.cat(features_list, dim=0)[:n_samples_needed]
        print(f"Processed {self.real_features.shape[0]} real images")
    
    def get_inception_features(self, images):
        """Extract features from inception model."""
        # Rescale from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        
        self.inception_model.eval()
        with torch.no_grad():
            features = self.inception_model(images)[0]
        
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features
    
    def generate_samples(self):
        """Generate samples using DDIM sampling."""
        from ldm.models.diffusion.ddim import DDIMSampler
        
        ddim_sampler = DDIMSampler(self.model)
        
        all_samples = []
        n_rounds = self.num_samples // self.batch_size
        
        with torch.no_grad():
            for _ in tqdm(range(n_rounds), desc="Generating samples"):
                shape = [self.batch_size, 
                        self.model.model.diffusion_model.in_channels, 
                        self.model.model.diffusion_model.image_size, 
                        self.model.model.diffusion_model.image_size]
                
                # 'steps' 대신 'S' 매개변수 사용
                samples, _ = ddim_sampler.sample(
                    S=self.ddim_steps,
                    batch_size=self.batch_size,
                    shape=shape[1:],
                    eta=self.ddim_eta,
                    verbose=False
                )
                x_samples = self.model.decode_first_stage(samples)
                all_samples.append(x_samples)
        
        return torch.cat(all_samples, dim=0)
    
    def calculate_inception_score(self, features, splits=10):
        """Calculate inception score."""
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
        """Calculate Fréchet Inception Distance."""
        mu1 = np.mean(fake_features.cpu().numpy(), axis=0)
        sigma1 = np.cov(fake_features.cpu().numpy(), rowvar=False)
        
        mu2 = np.mean(real_features.cpu().numpy(), axis=0)
        sigma2 = np.cov(real_features.cpu().numpy(), rowvar=False)
        
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    def save_results(self, results, save_samples=False, samples_dir='./generated_samples'):
        """Save evaluation results and optionally samples."""
        # Create results directory
        results_dir = './eval_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results dictionary
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
            'timestamp': timestamp
        }
        
        # Save results to JSON
        filename = os.path.join(results_dir, f'eval_results_{self.global_step}_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(output_dict, f, indent=4)
        
        print(f"\nResults saved to {filename}")
        
        # Save samples if requested
        if save_samples and 'samples' in results:
            os.makedirs(samples_dir, exist_ok=True)
            samples = results['samples']
            
            # Convert to PIL and save
            for i, sample in enumerate(samples[:100]):  # Save first 100 samples
                sample = (sample + 1) / 2  # [-1, 1] -> [0, 1]
                sample = sample.permute(1, 2, 0).cpu().numpy()
                sample = (sample * 255).astype(np.uint8)
                img = Image.fromarray(sample)
                img.save(os.path.join(samples_dir, f'sample_{i:05d}.png'))
            
            print(f"Sample images saved to {samples_dir}")
    
    def evaluate(self):
        """Run the full evaluation pipeline."""
        print(f"Starting evaluation for model at global step {self.global_step}")
        print(f"Generating {self.num_samples} samples...")
        
        # Generate samples
        samples = self.generate_samples()
        
        # Calculate Inception features for generated samples
        print("Calculating Inception features for generated samples...")
        all_fake_features = []
        
        with torch.no_grad():
            for i in range(0, samples.shape[0], self.batch_size):
                batch = samples[i:i+self.batch_size]
                features = self.get_inception_features(batch)
                all_fake_features.append(features)
        
        all_fake_features = torch.cat(all_fake_features, dim=0)
        
        # Calculate metrics
        print("Calculating Inception Score...")
        IS_score = self.calculate_inception_score(all_fake_features)
        print(f"Inception Score: {IS_score:.3f}")
        
        print("Calculating FID Score...")
        fid = self.calculate_fid(self.real_features, all_fake_features)
        print(f"FID Score: {fid:.3f}")
        
        # Prepare results
        results = {
            'inception_score': IS_score,
            'fid_score': fid,
            'samples': samples
        }
        
        # Save results
        self.save_results(results, save_samples=True)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Stable Diffusion model')
    parser.add_argument('--config', type=str, 
                       required=True,
                       help='Path to the model config file')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--dataset_path', type=str,
                       required=True,
                       help='Path to the real dataset for FID calculation')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0,
                       help='DDIM eta parameter (0.0 for deterministic)')
    
    args = parser.parse_args()
    
    evaluator = StableDiffusionEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta
    )
    
    results = evaluator.evaluate()
    
if __name__ == '__main__':
    main()