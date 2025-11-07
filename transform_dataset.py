"""
Dataset Transformer for HuggingFace Multilingual-Thinking format
Uses Claude via Amazon Bedrock with retry logic and concurrent processing.

Usage:
    # From HuggingFace Hub
    python transform_dataset.py HuggingFaceH4/Multilingual-Thinking output_dir --role user
    
    # From local directory
    python transform_dataset.py ./my_dataset output_dir --role user
    
    # From JSONL (legacy)
    python transform_dataset.py input.jsonl output_dir --role user --input-format jsonl
"""

import json
import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import signal

import boto3
from botocore.exceptions import ClientError
from datasets import load_dataset, Dataset
from tqdm import tqdm


@dataclass
class TransformConfig:
    """Configuration for dataset transformation"""
    message_role: str = "user"
    transform_thinking: bool = False
    keep_only_messages: bool = False
    max_concurrent: int = 10
    max_retries: int = 5
    model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str = "us-east-1"
    verbose: bool = False
    transform_prompt: str = (
        "Enhance this text by making it more detailed and specific, "
        "but keep the same core meaning. Return ONLY the enhanced text, no explanation.\n\n"
        "Original text: {content}"
    )


@dataclass
class TransformStats:
    """Statistics tracking for transformation"""
    success: int = 0
    errors: int = 0
    rate_limits: int = 0
    retries: int = 0
    active_requests: int = 0


class DatasetTransformer:
    """Transform HuggingFace datasets using Claude via Amazon Bedrock"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        self.stats = TransformStats()
        self.should_stop = False
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=config.aws_region
        )
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\n⚠️  Received interrupt signal. Stopping after current batch...")
        self.should_stop = True
    
    def _log(self, message: str, level: str = "INFO", verbose_only: bool = False):
        """Log a message with timestamp"""
        if verbose_only and not self.config.verbose:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        color_codes = {
            "INFO": "\033[0m",      # Default
            "SUCCESS": "\033[92m",  # Green
            "ERROR": "\033[91m",    # Red
            "WARNING": "\033[93m",  # Yellow
            "RETRY": "\033[33m",    # Orange
            "DEBUG": "\033[94m"     # Blue
        }
        color = color_codes.get(level, color_codes["INFO"])
        reset = "\033[0m"
        print(f"{color}[{timestamp}] {message}{reset}")
    
    async def _call_claude_with_retry(self, prompt: str) -> str:
        """Call Claude via Bedrock Converse API with retry logic"""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                self._log(f"Making API call (attempt {attempt + 1}/{self.config.max_retries})", "DEBUG", verbose_only=True)

                # Call Bedrock Converse API (run in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    lambda: self.bedrock.converse(
                        modelId=self.config.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"text": prompt}
                                ]
                            }
                        ],
                        inferenceConfig={
                            "maxTokens": 2000
                        }
                    )
                )

                self._log("API call succeeded", "DEBUG", verbose_only=True)

                # Extract text from response
                return response['output']['message']['content'][0]['text']

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')

                # Handle throttling
                if error_code == 'ThrottlingException':
                    self.stats.rate_limits += 1
                    self.stats.retries += 1

                    # Exponential backoff
                    wait_time = min(2 ** attempt, 32)
                    self._log(
                        f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retries}...",
                        "WARNING"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # Other errors
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = min(2 ** attempt, 16)
                    self.stats.retries += 1
                    self._log(
                        f"Request failed: {error_code}. Retrying in {wait_time}s ({attempt + 1}/{self.config.max_retries})...",
                        "RETRY"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = min(2 ** attempt, 16)
                    self.stats.retries += 1
                    self._log(
                        f"Request failed: {str(e)}. Retrying in {wait_time}s ({attempt + 1}/{self.config.max_retries})...",
                        "RETRY"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise last_error or Exception("Max retries exceeded")
    
    async def _transform_example(self, example: Dict, index: int, semaphore: asyncio.Semaphore, pbar=None) -> Dict:
        """Transform a single example"""
        async with semaphore:
            self.stats.active_requests += 1

            try:
                # Validate messages structure
                messages = example.get('messages')
                if not messages or not isinstance(messages, list):
                    self.stats.errors += 1
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': self.stats.success,
                            'errors': self.stats.errors,
                            'rate_limits': self.stats.rate_limits,
                            'retries': self.stats.retries,
                            'active': self.stats.active_requests
                        })
                    return {**example, '_transform_error': 'Missing messages array'}

                # Find message to transform
                message_index = next(
                    (i for i, msg in enumerate(messages) if msg.get('role') == self.config.message_role),
                    None
                )

                if message_index is None:
                    self.stats.errors += 1
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': self.stats.success,
                            'errors': self.stats.errors,
                            'rate_limits': self.stats.rate_limits,
                            'retries': self.stats.retries,
                            'active': self.stats.active_requests
                        })
                    return {**example, '_transform_error': f'No {self.config.message_role} message'}

                message = messages[message_index]
                content = message.get('content', '')
                thinking = message.get('thinking')

                # Transform content
                content_prompt = self.config.transform_prompt.replace('{content}', content)
                transformed_content = await self._call_claude_with_retry(content_prompt)

                # Transform thinking if enabled
                transformed_thinking = thinking
                if self.config.transform_thinking and self.config.message_role == 'assistant' and thinking:
                    thinking_prompt = self.config.transform_prompt.replace('{content}', thinking)
                    transformed_thinking = await self._call_claude_with_retry(thinking_prompt)

                # Update message
                new_messages = messages.copy()
                new_messages[message_index] = {
                    **message,
                    'content': transformed_content,
                    'thinking': transformed_thinking
                }

                result = {**example, 'messages': new_messages}
                self.stats.success += 1

                # Update progress bar
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': self.stats.success,
                        'errors': self.stats.errors,
                        'rate_limits': self.stats.rate_limits,
                        'retries': self.stats.retries,
                        'active': self.stats.active_requests
                    })

                return result

            except Exception as e:
                self.stats.errors += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': self.stats.success,
                        'errors': self.stats.errors,
                        'rate_limits': self.stats.rate_limits,
                        'retries': self.stats.retries,
                        'active': self.stats.active_requests
                    })
                return {**example, '_transform_error': str(e)}

            finally:
                self.stats.active_requests -= 1
    
    async def _transform_batch(self, examples: List[Dict], pbar=None) -> List[Dict]:
        """Transform a batch of examples concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        tasks = [
            self._transform_example(example, i, semaphore, pbar)
            for i, example in enumerate(examples)
        ]

        return await asyncio.gather(*tasks)
    
    def transform_dataset(self, dataset: Dataset, batch_size: int = 100) -> Dataset:
        """
        Transform entire dataset using async batch processing.
        
        Args:
            dataset: HuggingFace Dataset to transform
            batch_size: Number of examples to process in each batch (for progress tracking)
        
        Returns:
            Transformed Dataset
        """
        self._log(f"Starting transformation of {len(dataset)} examples...", "INFO")
        self._log(f"Transforming '{self.config.message_role}' messages", "INFO")
        self._log(f"Max concurrent requests: {self.config.max_concurrent}", "INFO")
        
        all_results = []
        total = len(dataset)
        
        # Process in batches for progress tracking
        with tqdm(total=total, desc="Transforming", unit="ex") as pbar:
            for i in range(0, total, batch_size):
                if self.should_stop:
                    self._log("Stopping as requested...", "WARNING")
                    break

                # Get batch
                batch_end = min(i + batch_size, total)
                batch = [dataset[j] for j in range(i, batch_end)]

                # Process batch (progress bar is updated inside _transform_example)
                batch_results = asyncio.run(self._transform_batch(batch, pbar))
                all_results.extend(batch_results)
        
        # Create new dataset from results
        if not all_results:
            self._log("No results to save (stopped early)", "WARNING")
            return dataset.select([])  # Return empty dataset
        
        # Apply keep_only_messages filter
        if self.config.keep_only_messages:
            all_results = [{'messages': item['messages']} for item in all_results]
            self._log("Kept only 'messages' field, removed all other columns", "INFO")
        
        transformed_dataset = Dataset.from_list(all_results)
        
        # Print final stats
        self._log("\n" + "="*50, "INFO")
        self._log("Transformation Complete!", "SUCCESS")
        self._log(f"Total processed: {len(all_results)}", "INFO")
        self._log(f"Success: {self.stats.success}", "SUCCESS")
        self._log(f"Errors: {self.stats.errors}", "ERROR")
        self._log(f"Rate limits hit: {self.stats.rate_limits}", "WARNING")
        self._log(f"Total retries: {self.stats.retries}", "RETRY")
        self._log("="*50 + "\n", "INFO")
        
        return transformed_dataset


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Transform HuggingFace datasets using Claude via Amazon Bedrock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace Hub
  python transform_dataset.py HuggingFaceH4/Multilingual-Thinking ./output --role user

  # From local dataset directory
  python transform_dataset.py ./my_dataset ./output --role assistant --transform-thinking

  # From JSONL file
  python transform_dataset.py data.jsonl ./output --role user --input-format jsonl

  # Push directly to Hub
  python transform_dataset.py HuggingFaceH4/Multilingual-Thinking username/my-dataset \\
      --role user --push-to-hub
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input dataset (HF Hub path, local dir, or JSONL file)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path (directory for save_to_disk, or HF Hub path with --push-to-hub)"
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
        help="Dataset split to transform (default: train)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push output directly to HuggingFace Hub (output should be a repo path)"
    )
    
    # Transformation options
    parser.add_argument(
        "--role",
        choices=["system", "user", "assistant"],
        default="user",
        help="Message role to transform (default: user)"
    )
    parser.add_argument(
        "--transform-thinking",
        action="store_true",
        help="Also transform the 'thinking' field for assistant messages"
    )
    parser.add_argument(
        "--keep-only-messages",
        action="store_true",
        help="Keep only 'messages' field, remove all other columns"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom transformation prompt (use {content} as placeholder)"
    )
    
    # Performance options
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=15,
        help="Maximum concurrent requests (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for progress tracking (default: 100)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    # AWS options
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model ID"
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
            dataset = load_dataset(args.input, split=args.split)
        else:  # hub
            dataset = load_dataset(args.input, split=args.split)
        
        print(f"✅ Loaded {len(dataset)} examples")
        print(f"📋 Features: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Create config
    config = TransformConfig(
        message_role=args.role,
        transform_thinking=args.transform_thinking,
        keep_only_messages=args.keep_only_messages,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        model_id=args.model_id,
        aws_region=args.region,
        verbose=args.verbose
    )
    
    if args.prompt:
        config.transform_prompt = args.prompt
    
    # Create transformer and run
    transformer = DatasetTransformer(config)
    
    try:
        transformed_dataset = transformer.transform_dataset(dataset, batch_size=args.batch_size)
        
        # Save output
        if args.push_to_hub:
            print(f"\n📤 Pushing to HuggingFace Hub: {args.output}...")
            transformed_dataset.push_to_hub(args.output)
            print(f"✅ Pushed to https://huggingface.co/datasets/{args.output}")
        else:
            print(f"\n💾 Saving to {args.output}...")
            transformed_dataset.save_to_disk(args.output)
            print(f"✅ Saved to {args.output}")
            
            # Also save as JSONL for convenience
            jsonl_path = Path(args.output) / "data.jsonl"
            with open(jsonl_path, 'w') as f:
                for item in transformed_dataset:
                    f.write(json.dumps(item) + '\n')
            print(f"✅ Also saved as {jsonl_path}")
        
        print("\n🎉 All done!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
