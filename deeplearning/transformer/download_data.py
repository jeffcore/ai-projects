import os
import argparse
import time
from datasets import load_dataset

def download_wikitext(output_dir, dataset_name="wikitext", subset="wikitext-2-v1", split="train"):
    """
    Download the WikiText dataset and save it to disk.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset
        subset: Subset of the dataset
        split: Split of the dataset (train, validation, test)
        
    Returns:
        Path to the saved dataset file
    """
    print(f"Downloading {dataset_name}/{subset} ({split} split)...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(dataset_name, subset, split=split)
    
    # Save the dataset to disk
    output_file = os.path.join(output_dir, f"{subset}-{split}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item["text"] + "\n")
    
    print(f"Dataset saved to {output_file}")
    return output_file

def download_generic_dataset(output_dir, dataset_name, text_field="text", subset=None, split="train"):
    """
    Download a generic text dataset from Hugging Face and save it to disk.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset on Hugging Face
        text_field: Name of the field containing the text in the dataset
        subset: Subset of the dataset (if applicable)
        split: Split of the dataset (train, validation, test)
        
    Returns:
        Path to the saved dataset file
    """
    print(f"Downloading {dataset_name} ({'with subset ' + subset if subset else 'no subset'}, {split} split)...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset with or without subset
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
            file_prefix = f"{dataset_name}-{subset}"
        else:
            dataset = load_dataset(dataset_name, split=split)
            file_prefix = dataset_name
        
        # Save the dataset to disk
        output_file = os.path.join(output_dir, f"{file_prefix}-{split}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                if text_field in item and item[text_field]:
                    f.write(item[text_field] + "\n")
        
        print(f"Dataset saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return None

def combine_datasets(output_dir, input_files, output_filename="combined_dataset.txt"):
    """
    Combine multiple dataset files into a single file.
    
    Args:
        output_dir: Directory to save the combined dataset
        input_files: List of input files to combine
        output_filename: Name of the output file
        
    Returns:
        Path to the combined dataset file
    """
    print(f"Combining {len(input_files)} datasets...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine the datasets
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, "w", encoding="utf-8") as outf:
        for input_file in input_files:
            if input_file and os.path.exists(input_file):
                print(f"Adding {input_file} to combined dataset")
                with open(input_file, "r", encoding="utf-8") as inf:
                    outf.write(inf.read())
                outf.write("\n\n")
    
    print(f"Combined dataset saved to {output_file}")
    return output_file

def download_multiple_datasets(output_dir, dataset_configs):
    """
    Download multiple datasets based on configurations.
    
    Args:
        output_dir: Directory to save the datasets
        dataset_configs: List of dataset configurations
        
    Returns:
        List of paths to the saved dataset files
    """
    dataset_files = []
    
    for config in dataset_configs:
        dataset_type = config.get("type", "generic")
        
        if dataset_type == "wikitext":
            file_path = download_wikitext(
                output_dir,
                config.get("name", "wikitext"),
                config.get("subset", "wikitext-2-v1"),
                config.get("split", "train")
            )
        else:  # generic dataset
            file_path = download_generic_dataset(
                output_dir,
                config.get("name"),
                config.get("text_field", "text"),
                config.get("subset"),
                config.get("split", "train")
            )
        
        if file_path:
            dataset_files.append(file_path)
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    return dataset_files

def main():
    parser = argparse.ArgumentParser(description="Download text datasets for transformer training")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the datasets")
    parser.add_argument("--combine", action="store_true", help="Combine all downloaded datasets into a single file")
    parser.add_argument("--dataset", type=str, default="all", 
                        choices=["all", "wikitext-2", "wikitext-103", "bookcorpus", "openwebtext", "pile-small"],
                        help="Preset dataset configuration to download")
    
    args = parser.parse_args()
    
    # Define preset dataset configurations
    preset_configs = {
        "wikitext-2": [
            {"type": "wikitext", "name": "wikitext", "subset": "wikitext-2-v1", "split": "train"}
        ],
        "wikitext-103": [
            {"type": "wikitext", "name": "wikitext", "subset": "wikitext-103-v1", "split": "train"}
        ],
        "bookcorpus": [
            {"name": "bookcorpus", "text_field": "text", "split": "train"}
        ],
        "openwebtext": [
            {"name": "Skylion007/openwebtext", "text_field": "text", "split": "train[:10000]"}
        ],
        "pile-small": [
            {"name": "EleutherAI/pile-small", "text_field": "text", "split": "train[:5000]"}
        ],
        "all": [
            {"type": "wikitext", "name": "wikitext", "subset": "wikitext-103-v1", "split": "train"},
            {"name": "bookcorpus", "text_field": "text", "split": "train[:5000]"},
            {"name": "Skylion007/openwebtext", "text_field": "text", "split": "train[:5000]"}
        ]
    }
    
    # Get dataset configuration
    dataset_configs = preset_configs.get(args.dataset, preset_configs["wikitext-2"])
    
    # Download the datasets
    dataset_files = download_multiple_datasets(args.output_dir, dataset_configs)
    
    # Combine the datasets if requested
    if args.combine and dataset_files:
        combine_datasets(args.output_dir, dataset_files, f"combined_{args.dataset}_dataset.txt")

if __name__ == "__main__":
    main()
