import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import json
import shutil
import uuid
import numpy as np
from PIL import Image

# DINOv3 architecture configurations
# Maps the human-readable model size to its corresponding architecture name,
# local checkpoint filename, and output embedding dimension.
MODEL_CONFIGS = {
    'small': {'arch': 'dinov3_vits16', 'ckpt': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 'dim': 384},
    'base':  {'arch': 'dinov3_vitb16', 'ckpt': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 'dim': 768},
    'large': {'arch': 'dinov3_vitl16', 'ckpt': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', 'dim': 1024}
}

class FastDataset(torch.utils.data.Dataset):
    """
    Optimized PyTorch Dataset for high-speed batch inference.
    Handles dynamic resizing to Vision Transformer (ViT) compatible dimensions
    and applies standard ImageNet normalization.

    Args:
        folder (str): Path to the directory containing the images.
        max_size (int, optional): The target maximum resolution for the longest edge. Defaults to 672.
    """
    def __init__(self, folder, max_size=672):
        self.folder = folder
        self.max_size = max_size
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        """Returns the total number of valid image files in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads, resizes, and normalizes an image at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the normalized image tensor and its filename.
        """
        name = self.files[idx]
        img_path = os.path.join(self.folder, name)
        img = Image.open(img_path).convert('RGB')
        
        # Calculate resize ratio while maintaining aspect ratio
        w, h = img.size
        ratio = min(self.max_size / w, self.max_size / h)
        
        # Ensure dimensions are strictly multiples of 16 (required by ViT patch size)
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        return self.normalize(img_resized), name

class PatchLibrary:
    """
    Manages the persistent storage, vector concatenation, metadata, 
    and UI visual cache for user-selected semantic patches.

    Args:
        lib_path (str): The root directory where the library structure will be created/loaded.
    """
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.images_dir = os.path.join(lib_path, "images")
        self.heatmaps_dir = os.path.join(lib_path, "heatmaps") # Cache folder for UI gallery previews
        self.metadata_path = os.path.join(lib_path, "metadata.json")
        self.vectors_path = os.path.join(lib_path, "vectors.pt")
        
        # Initialize directory structure if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        
        self.metadata = []
        self.vectors = None
        
        # Auto-load existing data if present
        if os.path.exists(self.metadata_path):
            self.load()

    def add_patch(self, vector, source_img_path, coords, dino_version, input_size, heatmap_pixmap):
        """
        Registers a new semantic patch into the library.

        Args:
            vector (torch.Tensor): The extracted DINOv3 feature vector for the patch.
            source_img_path (str): Filepath to the original image.
            coords (tuple): The (y, x) patch coordinates in the feature grid.
            dino_version (str): The architecture used (e.g., 'small').
            input_size (int): The resolution used during extraction.
            heatmap_pixmap (QPixmap): Pre-rendered UI heatmap to cache.
        """
        patch_uid = str(uuid.uuid4())[:8]
        file_name = os.path.basename(source_img_path)
        
        # 1. Save Heatmap rendering to cache (PNG) for instant UI loading
        hm_name = f"hm_{patch_uid}.png"
        heatmap_pixmap.save(os.path.join(self.heatmaps_dir, hm_name), "PNG")
        
        # 2. Archive a copy of the original source image
        dest_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(dest_path):
            shutil.copy2(source_img_path, dest_path)
        
        # 3. Append configuration to metadata list
        self.metadata.append({
            "image_name": file_name,
            "heatmap_cache": hm_name,
            "coords": coords,
            "dino_version": dino_version,
            "input_size": input_size
        })
        
        # 4. Concatenate the new vector to the main library tensor
        new_vec = vector.detach().cpu().reshape(1, -1)
        self.vectors = new_vec if self.vectors is None else torch.cat([self.vectors, new_vec], dim=0)
        
        self.save()

    def remove_patch(self, index):
        """
        Removes a patch from the library and cleans up its associated cached files.

        Args:
            index (int): The index of the patch to remove.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if self.vectors is not None and 0 <= index < len(self.metadata):
            # Clean up cached heatmap PNG
            hm_name = self.metadata[index].get("heatmap_cache")
            if hm_name:
                hm_path = os.path.join(self.heatmaps_dir, hm_name)
                if os.path.exists(hm_path): 
                    os.remove(hm_path)
                
            self.metadata.pop(index)
            
            # Rebuild vector tensor or delete file if empty
            if len(self.metadata) == 0:
                self.vectors = None
                if os.path.exists(self.vectors_path): 
                    os.remove(self.vectors_path)
            else:
                self.vectors = torch.cat([self.vectors[:index], self.vectors[index+1:]])
                
            self.save()
            return True
        return False

    def save(self):
        """Serializes the vector tensor (.pt) and metadata (.json) to disk."""
        if self.vectors is not None:
            torch.save(self.vectors, self.vectors_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def load(self):
        """Deserializes the vector tensor and metadata from disk into memory."""
        if os.path.exists(self.vectors_path):
            self.vectors = torch.load(self.vectors_path, map_location='cpu')
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    @staticmethod
    def merge(path_a, path_b, path_out):
        """
        Statically merges two independent libraries into a new destination folder.
        Combines metadata, concatenates vector tensors, and migrates all physical files.

        Args:
            path_a (str): Directory of the first source library.
            path_b (str): Directory of the second source library.
            path_out (str): Directory for the new combined library.

        Returns:
            PatchLibrary: An instance of the newly merged library.
        """
        lib_a, lib_b = PatchLibrary(path_a), PatchLibrary(path_b)
        lib_out = PatchLibrary(path_out)
        
        lib_out.metadata = lib_a.metadata + lib_b.metadata
        
        # Concatenate tensors if they exist
        vecs = [v for v in [lib_a.vectors, lib_b.vectors] if v is not None]
        if vecs:
            lib_out.vectors = torch.cat(vecs, dim=0)
            
        # Migrate all physical files (raw images and heatmap caches)
        for lib in [lib_a, lib_b]:
            for img in os.listdir(lib.images_dir):
                shutil.copy2(os.path.join(lib.images_dir, img), os.path.join(lib_out.images_dir, img))
            for hm in os.listdir(lib.heatmaps_dir):
                shutil.copy2(os.path.join(lib.heatmaps_dir, hm), os.path.join(lib_out.heatmaps_dir, hm))
        
        lib_out.save()
        return lib_out

class DinoManager:
    """
    Core Inference Engine wrapper for PyTorch Hub integration.
    Manages loading the architecture, moving weights to GPU/CPU, 
    and extracting normalized feature maps.

    Args:
        repo_dir (str): Absolute path to the local DINOv3 repository clone.
        model_size (str, optional): The initial architecture to load ('small', 'base', 'large').
    """
    def __init__(self, repo_url, model_size='small'):
        self.repo_url = repo_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_config = None
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # Tracks the active model size for external querying (e.g., UI status bar)
        self.current_size = model_size

    def load_model(self, size):
        """
        Loads the DINOv3 model automatically from GitHub.
        If already downloaded, Torch will use the local cache.
        """
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.current_size = size
        self.current_config = MODEL_CONFIGS[size]

        # --- LA MODIFICATION CLEF ---
        # source='github' : Va chercher sur le web
        # pretrained=True : Télécharge et charge les poids .pth automatiquement
        self.model = torch.hub.load(
            repo_or_dir=self.repo_url, 
            model=self.current_config['arch'], 
            source='github',
            pretrained=True
        )

        self.model.to(self.device).eval()

    @torch.inference_mode()
    def get_features(self, pil_img, max_size=672):
        """
        Processes a raw PIL Image and extracts its dense semantic feature map.

        Args:
            pil_img (PIL.Image): The input image object.
            max_size (int, optional): The maximum edge resolution before resizing.

        Returns:
            tuple: Contains the normalized feature tensor and a tuple of the final computed dimensions (width, height).
        """
        w, h = pil_img.size
        ratio = min(max_size / w, max_size / h)
        
        # Enforce ViT 16x16 patch constraint
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        img_t = self.transform(pil_img.resize((new_w, new_h), Image.BILINEAR)).unsqueeze(0).to(self.device)
        
        # Utilize automatic mixed precision for accelerated inference on supported hardware
        with torch.amp.autocast(device_type=self.device.type):
            # Extract raw patches from the final intermediate layer
            feat = self.model.get_intermediate_layers(img_t, n=1)[0]
            hp, wp = new_h // 16, new_w // 16
            
            # Reshape flat sequence back into 2D spatial grid and apply L2 normalization
            feat = F.normalize(feat[0, -(hp*wp):, :].reshape(hp, wp, -1), dim=-1)
            
        return feat, (new_w, new_h)

class AppState:
    """
    Centralized state container for the Model-View-Controller architecture.
    Maintains references to the AI engine, the loaded library, and the active dataset.

    Args:
        repo_dir (str): Path to the DINOv3 repository for engine initialization.
    """
    def __init__(self, repo_dir):
        self.dino = DinoManager(repo_dir)
        self.active_library = None
        self.image_folder = None
        self.image_list = []
        self.current_image_idx = 0