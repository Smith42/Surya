import yaml
import torch
import numpy as np
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from surya.datasets.helio import HelioNetCDFDataset
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn


def download_model_data():
    """Download model weights and configuration from HuggingFace."""
    snapshot_download(
        repo_id="nasa-ibm-ai4science/Surya-1.0",
        local_dir="data/Surya-1.0",
        allow_patterns=["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
    )


def load_config(config_path="data/Surya-1.0/config.yaml"):
    """Load model configuration."""
    with open(config_path) as fp:
        return yaml.safe_load(fp)


def load_scalers(scalers_path="data/Surya-1.0/scalers.yaml"):
    """Build scalers for data normalization."""
    scalers_info = yaml.safe_load(open(scalers_path, "r"))
    return build_scalers(info=scalers_info)


def create_model(config):
    """Initialize the HelioSpectFormer model."""
    model = HelioSpectFormer(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=len(config["data"]["sdo_channels"]),
        embed_dim=config["model"]["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": len(config["data"]["time_delta_input_minutes"])
        },
        depth=config["model"]["depth"],
        n_spectral_blocks=config["model"]["n_spectral_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=torch.bfloat16,
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["model"]["learned_flow"],
        init_weights=False,
        checkpoint_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        finetune=config["model"]["finetune"],
    )
    return model


def load_model_weights(model, weights_path, device):
    """Load pretrained weights into the model."""
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model


def create_dataset(config, scalers, index_path):
    """Create the dataset for inference."""
    dataset = HelioNetCDFDataset(
        index_path=index_path,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
        rollout_steps=1,
        channels=config["data"]["sdo_channels"],
        drop_hmi_probability=0.0,
        num_mask_aia_channels=0,
        use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        pooling=config["data"]["pooling"],
        random_vert_flip=False,
    )
    return dataset


def run_inference(model, dataloader, device):
    """Run inference on the dataloader."""
    model.eval()
    results = []
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           dtype=torch.bfloat16):
            for batch_idx, (batch_data, batch_metadata) in enumerate(dataloader):
                x = batch["ts"]
                dt = batch["time_delta_input"]
                
                tokens = model.embedding(x, dt)
                embeddings = model.backbone(tokens)
                
                print(f"Batch {batch_idx}: Embeddings shape: {embeddings.shape}")
                results.append({
                    "embeddings": embeddings.cpu(),
                    "metadata": batch_metadata
                })
    
    return results


def main():
    """Main inference pipeline."""
    # Download data (comment out if already downloaded)
    print("Downloading model and configuration...")
    download_model_data()
    
    # Load configuration and scalers
    print("Loading configuration and scalers...")
    config = load_config()
    scalers = load_scalers()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create and load model
    print("Initializing model...")
    model = create_model(config)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_parameters:.2f}M")
    
    print("Loading pretrained weights...")
    model = load_model_weights(model, "data/Surya-1.0/surya.366m.v1.pt", device)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    index_path = "path/to/your/index.csv"  # Update this path
    dataset = create_dataset(config, scalers, index_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=2,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, dataloader, device)
    
    print(f"Inference complete! Processed {len(results)} batches.")
    

if __name__ == "__main__":
    main()
