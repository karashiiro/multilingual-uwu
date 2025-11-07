"""
Push a local HuggingFace dataset to the Hub.

Usage:
    # Push a dataset saved with save_to_disk
    python push_to_hub.py ./output_dir username/dataset-name

    # Push with custom split
    python push_to_hub.py ./output_dir username/dataset-name --split train

    # Preview dataset before pushing
    python push_to_hub.py ./output_dir username/dataset-name --preview

    # Push with README metadata
    python push_to_hub.py ./output_dir username/dataset-name --description "My dataset" --license mit
"""

import argparse
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset
from huggingface_hub import login


def preview_dataset(dataset: Dataset, num_examples: int = 3):
    """Preview dataset contents"""
    print("\n" + "="*60)
    print("📊 DATASET PREVIEW")
    print("="*60)
    print(f"Total examples: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")
    print(f"\nShowing first {min(num_examples, len(dataset))} examples:\n")

    for i, example in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
        print(f"\n--- Example {i+1} ---")
        for key, value in example.items():
            if key == 'messages' and isinstance(value, list):
                print(f"{key}:")
                for j, msg in enumerate(value):
                    print(f"  [{j}] {msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:100]}...")
                    if 'thinking' in msg and msg['thinking']:
                        print(f"      thinking: {str(msg['thinking'])[:100]}...")
            else:
                print(f"{key}: {str(value)[:200]}")
    print("\n" + "="*60)


def create_readme(repo_id: str, description: str = None, license: str = None):
    """Create a basic README for the dataset"""
    readme = f"""# {repo_id.split('/')[-1]}

{description or 'This dataset was created using the multilingual-uwu transformer.'}

## Dataset Structure

This dataset contains conversational data in the HuggingFace format with a `messages` field.

### Data Fields

- `messages`: A list of message dictionaries, each containing:
  - `role`: The role of the message sender (system, user, or assistant)
  - `content`: The message content
  - `thinking`: (optional) Extended thinking content for assistant messages

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Example: Access first conversation
print(dataset['train'][0]['messages'])
```

## Creation

This dataset was created using the [multilingual-uwu](https://github.com/karashiiro/multilingual-uwu) transformer script.

"""

    if license:
        readme += f"\n## License\n\n{license}\n"

    return readme


def main():
    parser = argparse.ArgumentParser(
        description="Push a local HuggingFace dataset to the Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic push
  python push_to_hub.py ./output username/my-dataset

  # Preview before pushing
  python push_to_hub.py ./output username/my-dataset --preview

  # With metadata
  python push_to_hub.py ./output username/my-dataset \\
      --description "My transformed dataset" \\
      --license mit \\
      --private
        """
    )

    # Required arguments
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to local dataset directory (created by save_to_disk)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace Hub repository ID (format: username/dataset-name)"
    )

    # Optional arguments
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview dataset before pushing"
    )
    parser.add_argument(
        "--num-preview",
        type=int,
        default=3,
        help="Number of examples to preview (default: 3)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Dataset description for README"
    )
    parser.add_argument(
        "--license",
        type=str,
        help="Dataset license (e.g., mit, apache-2.0, cc-by-4.0)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (if not already logged in)"
    )
    parser.add_argument(
        "--create-readme",
        action="store_true",
        help="Create a basic README.md for the dataset"
    )

    args = parser.parse_args()

    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if not dataset_path.is_dir():
        print(f"❌ Error: Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    # Validate repo_id format
    if '/' not in args.repo_id:
        print(f"❌ Error: repo_id must be in format 'username/dataset-name'")
        sys.exit(1)

    # Login to HuggingFace if token provided
    if args.token:
        print("🔐 Logging in to HuggingFace...")
        login(token=args.token)

    # Load dataset
    print(f"📂 Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(str(dataset_path))
        print(f"✅ Loaded {len(dataset)} examples")
        print(f"📋 Features: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Preview if requested
    if args.preview:
        preview_dataset(dataset, args.num_preview)
        response = input("\n📤 Continue with push? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            sys.exit(0)

    # Create README if requested
    if args.create_readme:
        print("📝 Creating README.md...")
        readme_content = create_readme(args.repo_id, args.description, args.license)
        readme_path = dataset_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Created {readme_path}")

    # Push to hub
    print(f"\n📤 Pushing to HuggingFace Hub: {args.repo_id}...")
    print(f"   Split: {args.split}")
    print(f"   Private: {args.private}")

    try:
        dataset.push_to_hub(
            args.repo_id,
            split=args.split,
            private=args.private
        )

        print(f"\n✅ Successfully pushed to Hub!")
        print(f"🔗 View at: https://huggingface.co/datasets/{args.repo_id}")

    except Exception as e:
        print(f"\n❌ Error pushing to Hub: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged in: huggingface-cli login")
        print("  2. Check that your token has write permissions")
        print("  3. Verify the repo_id format is correct: username/dataset-name")
        sys.exit(1)


if __name__ == "__main__":
    main()
