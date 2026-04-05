import os
import torch
import multiprocessing
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torchinfo import summary
import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gc
from src.task2 import logger
from src.task2.entity.config import ModelBuildConfig

class ModelBuild:
    def __init__(self, config:ModelBuildConfig ):
        self.config=config
        self.ps= self.config.patch_size
        self.grid_size  = self.config.input_size// self.config.patch_size
        self.n_token   = self.grid_size ** 2
        
    def sysConfig(self)-> str:
        logger.info("🔍 --- Project Diagnostic ---")

        # 1. Multiprocessing Check
        # Colab uses Linux, which defaults to 'fork'. 
        # For CUDA, 'spawn' is technically safer to avoid deadlocks.
        logger.info("[MULTIPROCESSING]")
        try:
            start_method = multiprocessing.get_start_method()
            logger.info(f"  Start method : {start_method}")
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"  CPU cores    : {cpu_count} logical")
        except Exception as e:
            logger.error(f"  Multiprocessing check failed: {e}")

        # 2. Hardware & Backend Check
        logger.info("[DEVICE & BACKEND]")
        
        # NVIDIA CUDA (Colab Standard)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            prop = torch.cuda.get_device_properties(0)
            
            logger.info(f"  Backend      ⚙️ : CUDA (NVIDIA)")
            logger.info(f"  GPU Model    : {prop.name}")
            logger.info(f"  VRAM Total   : {prop.total_memory / 1e9:.1f} GB")
            
            # Check for 'Compute Capability' (DINOv2 runs best on 7.0+)
            cc = f"{prop.major}.{prop.minor}"
            logger.info(f"  Compute Cap  : {cc}")
            
            # Check Memory Fragmentation
            vram_reserved = torch.cuda.memory_reserved(0) / 1e9
            vram_allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"  VRAM Reserved: {vram_reserved:.1f} GB")
            logger.info(f"  VRAM Active  : {vram_allocated:.1f} GB")

            # Smoke Test
            try:
                test_tensor = torch.zeros((100, 100), device=device)
                logger.info("  Smoke test 💨 : CUDA Tensor creation OK")
                del test_tensor # Clean up immediately
            except Exception as e:
                logger.warning(f"  Smoke test 💨 : FAILED — {e}")

        # Apple Silicon (For when you run locally)
        elif torch.backends.mps.is_available():
            device= torch.device('mps')
            logger.info("  Backend      ⚙️ : Apple Silicon (MPS)")
            logger.info("  Status       ✅: OK")

        else:
            logger.warning("  Backend      ⚙️ : CPU only — No Accelerator Found")

        # 3. Environment Context (Colab Specific)
        if 'COLAB_GPU' in os.environ:
            logger.info("  Environment  🌐: Google Colab detected")

        logger.info(f'Loading device and storing: {device}')
        
        return device

    def loadFromHuggingFace(self,device: str,name='model_summary_hf.txt'):
        model = AutoModel.from_pretrained(
            self.config.hug_mw, 
            cache_dir=self.config.HF_save,
            device_map="auto"
        )
        processor= AutoImageProcessor.from_pretrained(self.config.hug_mw)

        model_stats = summary(
            model, 
            input_size=tuple(self.config.param_dd),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            col_width=20,
            device=device
            )
        save=os.path.join(self.config.out,name)

        with open(save, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Analysis Run: {timestamp} ---\n Hugging Face Loaded \n")
            f.write(f"Model Summary for {self.config.model_name}\n")
            f.write(f"-"*30+"\n")
            f.write(f"{model_stats}\n")
            f.write(f"-"*30+"\n")
            f.write(f"==== Model BluePrint ====\n")
            f.write(f"{model}\n")
            f.write(f"-"*30+"\n")
            f.write(f"-"*30+"\n")
            f.write(f"ImageNet mean: {processor.image_mean}")
            f.write(f"ImageNet std: {processor.image_std}")
            f.write(f"-"*30+"\n")
            
        backbone='HF'

        return model, backbone
    
    def loadFromLocal(self,device:str, name='model_summary.txt'):
        model = torch.hub.load(
            self.config.repo, 
            self.config.model_name, 
            pretrained=False
        ).to(device)

        checkpoint = torch.load(self.config.mw, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        

        logger.info(f"DINOv3 {self.config.model_name} Weights successfully injected from flat checkpoint.")


        model_stats = summary(
            model,
            input_size=tuple(self.config.param_dd),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            col_width=20,
            device=device
            )
        save=os.path.join(self.config.out,name)

        # checkpoint = torch.load(self.config.mw, map_location=device)
        with open(save, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Analysis Run: {timestamp} ---\n")
            f.write(f"==== Model BluePrint LOCAL LOAD ====\n")
            f.write(f"{model}\n")
            f.write(f"-"*30+"\n")
            f.write(f"Model Summary for {self.config.model_name}\n")
            f.write(f"-"*30+"\n")
            f.write(f"{model_stats}\n")
            f.write(f"-"*30+"\n")
        
        backbone='Local'

        return model.to(device).eval(),backbone
    
    def _normalise(self,chunk: np.ndarray, device:str  ) -> torch.Tensor:
        # 1. Convert to float and scale to [0, 1]
        x = torch.from_numpy(chunk.astype(np.float32)).to(device)
        # Per-slice min-max normalisation → [0, 1]
        B = x.shape[0]
        mn = x.flatten(1).min(1).values.view(B, 1, 1)
        mx = x.flatten(1).max(1).values.view(B, 1, 1)
        x  = (x - mn) / (mx - mn + 1e-6)
        # 2. Stack to 3 channels: (Batch, 1, 448, 448) -> (Batch, 3, 448, 448)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        # 3. Apply ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        x = (x - mean) / std
        return x
    
    def _extract_patch_tokens(self, model, rgb: torch.Tensor,backbone: str) -> torch.Tensor:
        """
        rgb     : [B, 3, 448, 448]
        returns : [B, 784, 384]   pure patch tokens, registers stripped
        """
        if backbone == 'Local':
            out = model.forward_features(rgb)

            # Try the standard key first
            for key in ('x_norm_patchtokens', 'patch_tokens', 'x_patch_tokens'):
                if key in out:
                    tokens = out[key]           # [B, 784, 384] — already strips CLS+registers
                    break
                else:
                    
                    raise KeyError(
                        f'Patch token key not found.\n'
                        f'Available keys : {list(out.keys())}\n'
                        f'Hint: print out["x_norm_patchtokens"].shape to verify'
                    )
        if backbone == 'HF':
            out = model(rgb)
            full_sequence = out.last_hidden_state
            # We skip the first 5 tokens (1 CLS + 4 Registers)
            tokens = full_sequence[:, 5:, :]

        # Verify shape matches your model summary
        B, N, C = tokens.shape
        assert N == self.n_token,   f'Expected {self.n_token} patch tokens, got {N}'
        assert C == self.config.embed_dim,  f'Expected embed_dim={self.config.embed_dim}, got {C}'

        return tokens                       # [B, 784, 384]
    

    def get_dense_embeddings(self,device: str,model,backbone: str,file):
        fl= os.path.join(self.config.localfile,file)
        volume =np.load(fl)
        limit_slices= self.config.slide_use

        assert volume.ndim == 3, f'Expected [D,H,W], got {volume.shape}'
        D, H, W = volume.shape
        assert H == W == self.config.input_size, f'Expected {self.config.input_size}x{self.config.input_size}, got {H}x{W}'
        stop_at = min(limit_slices, D) if limit_slices else D
        working_volume = volume[:stop_at]

        logger.info(
            f'[{backbone}] {self.config.model_name} | '
            f'embed_dim={self.config.embed_dim} | '
            f'patch_size={self.config.patch_size} | '
            f'grid={self.grid_size}x{self.grid_size} | '
            f'registers={self.config.n_regis}'
        )
        all_slices = []
        with torch.no_grad():
            for i in range(0, stop_at, self.config.sli_batch):
                chunk = working_volume[i : i + self.config.sli_batch]    # [B, H, W]
                B     = chunk.shape[0]

                rgb          = self._normalise(chunk,device)      # [B, 3, 448, 448]
                patch_tokens = self._extract_patch_tokens(model,rgb,backbone)          # [B, 784, 384]

                # Reshape: flat sequence → 2D spatial grid
                grid = (patch_tokens
                        .permute(0, 2, 1)                               # [B, 384, 784]
                        .reshape(B, self.config.embed_dim,
                                self.grid_size, self.grid_size))       # [B, 384, 28, 28]

                # Upsample: 28×28 → 448×448
                dense = F.interpolate(
                    grid,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )                                                        # [B, 384, 448, 448]

                all_slices.append(dense.half().cpu())
                if device == "mps":
                    torch.mps.empty_cache()
                logger.info(
                f'  Slices {i:>4}-{min(i+B, stop_at)-1:<4} / {stop_at-1} | '
                f'[B={B}, C={self.config.embed_dim}, '
                f'{self.grid_size}x{self.grid_size}] → [{H}x{W}]'
            )
        logger.info("Merging all slices into 3D volume... (Please wait)")
        result = torch.cat(all_slices, dim=0).permute(1, 0, 2, 3)      # [384, D, 448, 448]
        name = input("Dense Embeddings| What should I name the output file? (e.g., dense.pt): ")
        n_sav= os.path.join(self.config.den_sav,name)
        torch.save(result, n_sav)
        logger.info(f"Saved 4D tensor to {n_sav}")
        logger.info(
            f'Complete | shape={list(result.shape)} | '
            f'range=[{result.min():.3f}, {result.max():.3f}]'
        )
        del result
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("Memory fully cleared.Mac is now fresh for the next task!")


    def visualize_embeddings_pca(self,slice_idx):
        """
        embeddings_3d: Tensor of shape [384, D, 448, 448]
        """
        
        # 1. Prepare the data for PCA
        # Flatten the 3D volume into a long list of 384-digit codes
        # [384, D, H, W] -> [D*H*W, 384]
        name= input("Dense Embeddings| What is the name of file? (e.g., dense.pt): ")
        n_sav= os.path.join(self.config.den_sav,name)
        embeddings_3d=torch.load(n_sav)
        file = input("Input Original file corresponding with dense embeddings?(e.g., liver.npy):")
        fl= os.path.join(self.config.localfile,file)
        original_volume = np.load(fl)
        C, D, H, W = embeddings_3d.shape
        if slice_idx is None:
            slice_idx = D // 2
        flat_embeddings = embeddings_3d.permute(1, 2, 3, 0).reshape(-1, C).numpy()

        # 2. Run PCA to find the 3 most important "Colors"
        # We sample 10,000 random points to make it fast on your M2 Pro
        logger.info("Computing PCA components...")
        pca = PCA(n_components=3)
        indices = np.random.choice(flat_embeddings.shape[0], 10000, replace=False)
        pca.fit(flat_embeddings[indices])

        # 3. Project the entire volume into RGB space
        pca_results = pca.transform(flat_embeddings) # Result is [Pixels, 3]
        
        # 4. Reshape back into a 3D Image
        # Result shape: [D, H, W, 3] (Ready for display)
        pca_volume = pca_results.reshape(D, H, W, 3)

        # 5. Normalise to [0, 1] for matplotlib
        pca_volume = (pca_volume - pca_volume.min()) / (pca_volume.max() - pca_volume.min())

        # 6. Show the middle slice
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(original_volume[slice_idx], cmap='gray')
        axes[0].set_title(f"Original EM (Slice {slice_idx})", fontsize=15)
        axes[0].axis('off')

        axes[1].imshow(pca_volume[slice_idx])
        axes[1].set_title(f"DINOv3 PCA Features (Slice {slice_idx})", fontsize=15)
        axes[1].axis('off')

        plt.tight_layout()
        save_name = input("Stage 2. What should I name the output file? (e.g., pca.png): ")
        save_path = os.path.join(self.config.out, save_name)
        plt.savefig(save_path, dpi=300) # Save high-res for your report
        logger.info(f"Comparison plot saved to {save_path}")
        plt.show()

        # Memory Cleanup
        del embeddings_3d, original_volume, flat_embeddings, pca_volume
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("Memory fully cleared.Mac is now fresh for the next task!")


