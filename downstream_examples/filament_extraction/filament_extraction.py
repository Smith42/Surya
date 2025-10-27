import yaml
import torch
import numpy as np
import os
import polars as pl
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from surya.datasets.helio import HelioNetCDFDataset
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn


def download_model_data():
    """Download model weights and configuration from HuggingFace."""
    print("Downloading Surya model weights and configuration...")
    snapshot_download(
        repo_id="nasa-ibm-ai4science/Surya-1.0",
        local_dir="data/Surya-1.0",
        allow_patterns=["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
    )
    print("Model download complete!")


def download_sdo_data():
    """Download SDO training data including index files and sample NetCDF data."""
    print("Downloading SDO training data...")
    
    # Download the index files and sample inference data
    snapshot_download(
        repo_id="nasa-ibm-ai4science/SDO_training",
        repo_type="dataset",
        local_dir="data/SDO_training",
        allow_patterns=["*_index_surya_1_0.csv", "infer_data/*", "scalers.yaml"],
    )
    
    print("SDO data download complete!")
    return "data/SDO_training"


def load_config(config_path="data/Surya-1.0/config.yaml"):
    """Load model configuration."""
    print(f"Loading configuration from {config_path}...")
    with open(config_path) as fp:
        config = yaml.safe_load(fp)
    print("Configuration loaded!")
    return config


def load_scalers(scalers_path="data/Surya-1.0/scalers.yaml"):
    """Build scalers for data normalization."""
    print(f"Loading scalers from {scalers_path}...")
    scalers_info = yaml.safe_load(open(scalers_path, "r"))
    scalers = build_scalers(info=scalers_info)
    print("Scalers loaded!")
    return scalers


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
    print("Model initialized!")
    return model


def load_model_weights(model, weights_path, device):
    """Load pretrained weights into the model."""
    print(f"Loading pretrained weights from {weights_path}...")
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)
    print("Weights loaded!")
    return model


def create_dataset(config, scalers, sdo_data_root, index_path):
    """Create the dataset for inference."""
    print(f"Creating dataset from {index_path}...")

    time_delta_target = config["data"]["time_delta_target_minutes"]
    if isinstance(time_delta_target, str):
        time_delta_target = int(time_delta_target.replace("+", ""))
    elif isinstance(time_delta_target, list):
        time_delta_target = time_delta_target[0]

    dataset = HelioNetCDFDataset(
        sdo_data_root_path=sdo_data_root,
        index_path=index_path,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=time_delta_target,
        n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
        rollout_steps=0,
        channels=config["data"]["sdo_channels"],
        drop_hmi_probability=0.0,
        num_mask_aia_channels=0,
        use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        pooling=config["data"]["pooling"],
        random_vert_flip=False,
    )
    print(f"Dataset created with {len(dataset)} samples!")
    return dataset


def run_inference(model, dataloader, device):
    """Run inference on the dataloader."""
    print("Running inference...")
    model.eval()
    results = []
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                           dtype=torch.bfloat16):
            for batch_idx, (batch_data, batch_metadata) in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch_data.items()}
                
                x = batch["ts"]
                dt = batch["time_delta_input"]
                
                # Get embeddings from the model
                tokens = model.embedding(x, dt)
                embeddings = model.backbone(tokens)
                
                print(f"  Batch {batch_idx + 1}: Embeddings shape: {embeddings.shape}")
                results.append({
                    "embeddings": embeddings.cpu(),
                    "metadata": batch_metadata,
                    "timestamps_input": batch_metadata.get("timestamps_input"),
                    "timestamps_targets": batch_metadata.get("timestamps_targets")
                })
    
    return results


def main():
    # Download data (comment out if already downloaded)
    print("Downloading model and configuration...")
    download_model_data()
    
    print("Downloading SDO data...")
    sdo_data_root = download_sdo_data()
    
    # Load configuration and scalers
    print("Loading configuration and scalers...")
    config = load_config()
    scalers = load_scalers()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create and load model
    print("\nInitializing model...")
    model = create_model(config)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_parameters:.2f}M")
    
    print("\nLoading pretrained weights...")
    model = load_model_weights(model, "data/Surya-1.0/surya.366m.v1.pt", device)
    
    print("\nCreating dataset...")
    index_path = os.path.join(sdo_data_root, "infer_data", "infer_index_surya_1_0.csv")
    
    if not os.path.exists(index_path):
        print(f"\nWarning: Index file not found at {index_path}")
        print("Available index files:")
        for root, dirs, files in os.walk(sdo_data_root):
            for file in files:
                if file.endswith('.csv'):
                    print(f"    - {os.path.join(root, file)}")
        raise FileNotFoundError(f"Could not find index file at {index_path}")
    
    dataset = create_dataset(config, scalers, os.path.join(sdo_data_root, "infer_data"), index_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    
    # Run inference
    print(f"Running inference on {len(dataset)} samples...")
    results = run_inference(model, dataloader, device)
    
    print(f"Inference complete! Processed {len(results)} batches.")
    
    # Display summary of results
    print("\nResults Summary:")
    for i, result in enumerate(results):
        print(f"\n  Sample {i+1}:")
        print(f"    Embedding shape: {result['embeddings'].shape}")
        if result.get('timestamps_input') is not None:
            print(f"    Input timestamps: {result['timestamps_input']}")
        if result.get('timestamps_targets') is not None:
            print(f"    Target timestamps: {result['timestamps_targets']}")
    
    return results
    

if __name__ == "__main__":
    data = main()
    df = pl.DataFrame(data)
    df.write_parquet("surya.parquet")
