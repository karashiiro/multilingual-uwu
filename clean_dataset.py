"""
Clean Dataset - Remove unnecessary fields and keep only 'messages'

Usage:
    # From saved dataset directory
    python clean_dataset.py input_dir output_dir

    # From JSONL file
    python clean_dataset.py input.jsonl output.jsonl --input-format jsonl

    # Push to Hub
    python clean_dataset.py input_dir username/repo --push-to-hub
"""

import json
import argparse
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset


def clean_dataset(dataset: Dataset) -> Dataset:
    """
    Remove all fields except 'messages' from the dataset.

    Args:
        dataset: Input dataset

    Returns:
        Cleaned dataset with only 'messages' field
    """
    print(f"📋 Original features: {list(dataset.features.keys())}")

    # Extract only messages field
    cleaned_data = [{'messages': item['messages']} for item in dataset]
    cleaned_dataset = Dataset.from_list(cleaned_data)

    print(f"✨ Cleaned features: {list(cleaned_dataset.features.keys())}")
    print(f"✅ Processed {len(cleaned_dataset)} examples")

    return cleaned_dataset


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Clean dataset by keeping only 'messages' field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From dataset directory
  python clean_dataset.py ./input_dir ./output_dir

  # From JSONL file
  python clean_dataset.py data.jsonl cleaned.jsonl --input-format jsonl

  # Push to Hub
  python clean_dataset.py ./input_dir username/my-dataset --push-to-hub
        """
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input dataset (directory, JSONL file, or HF Hub path)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path (directory for save_to_disk, JSONL file, or HF Hub path with --push-to-hub)"
    )

    # Input/Output options
    parser.add_argument(
        "--input-format",
        choices=["auto", "hub", "local", "jsonl"],
        default="auto",
        help="Input format (default: auto-detect)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push output directly to HuggingFace Hub (output should be a repo path)"
    )

    args = parser.parse_args()

    # Determine input format
    input_format = args.input_format
    if input_format == "auto":
        if args.input.endswith('.jsonl') or args.input.endswith('.json'):
            input_format = "jsonl"
        elif Path(args.input).is_dir():
            input_format = "local"
        else:
            input_format = "hub"

    # Load dataset
    print(f"📂 Loading dataset from {args.input}...")

    try:
        if input_format == "jsonl":
            dataset = load_dataset('json', data_files=args.input, split='train')
        elif input_format == "local":
            # Check if it's a saved dataset directory
            dataset_path = Path(args.input)
            if (dataset_path / "dataset_info.json").exists() or (dataset_path / "state.json").exists():
                dataset = load_from_disk(args.input)
            else:
                dataset = load_dataset(args.input, split=args.split)
        else:  # hub
            dataset = load_dataset(args.input, split=args.split)

        print(f"✅ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Clean dataset
    try:
        cleaned_dataset = clean_dataset(dataset)
    except Exception as e:
        print(f"❌ Error cleaning dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save output
    try:
        if args.push_to_hub:
            print(f"\n📤 Pushing to HuggingFace Hub: {args.output}...")
            cleaned_dataset.push_to_hub(args.output)
            print(f"✅ Pushed to https://huggingface.co/datasets/{args.output}")
        elif args.output.endswith('.jsonl') or args.output.endswith('.json'):
            # Save as JSONL
            print(f"\n💾 Saving to {args.output}...")
            with open(args.output, 'w') as f:
                for item in cleaned_dataset:
                    f.write(json.dumps(item) + '\n')
            print(f"✅ Saved to {args.output}")
        else:
            # Save as dataset directory
            print(f"\n💾 Saving to {args.output}...")
            cleaned_dataset.save_to_disk(args.output)
            print(f"✅ Saved to {args.output}")

            # Also save as JSONL for convenience
            jsonl_path = Path(args.output) / "data.jsonl"
            with open(jsonl_path, 'w') as f:
                for item in cleaned_dataset:
                    f.write(json.dumps(item) + '\n')
            print(f"✅ Also saved as {jsonl_path}")

        print("\n🎉 All done!")

    except Exception as e:
        print(f"\n❌ Error saving dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
