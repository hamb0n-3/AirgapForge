#!/usr/bin/env python3
import json
import os
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VulnerabilityDatasetConverter:
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_message = (
            "You are a advanced red team operator. Identify vulnerabilities, explain exploitability "
            "then provide step-by-step exploitation steps and payload examples."
        )
        
        self.stats = {
            'total_entries': 0,
            'converted_entries': 0,
            'failed_entries': 0,
            'format_used': '',
            'train_entries': 0,
            'valid_entries': 0,
            'test_entries': 0
        }
    
    def load_dataset_entries(self) -> List[Dict]:
        print("Loading dataset from Hugging Face...")
        try:
            ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
        
        entries = []
        for ex in ds:
            vulnerable_code = ex.get("rejected")
            vulnerability_description = ex.get("vulnerability")
            
            if vulnerable_code is None or vulnerability_description is None:
                self.stats['failed_entries'] += 1
                continue

            entry = {
                'user_message': f"Task: (identify the vulnerability and provide guidance for exploitation):\n\n{vulnerable_code}",
                'assistant_response': (
                    f"VULNERABILITY: {vulnerability_description}\n"
                    "EXPLOITATION STEPS:\n"
                    "- <step-by-step instruction on how to exploit the bug>\n\n"
                    "PAYLOADS:\n"
                    "- <Example payloads>\n\n"
                )
            }
            entries.append(entry)
            self.stats['total_entries'] += 1
        
        print(f"Loaded {len(entries)} valid entries from dataset")
        return entries
    
    def convert_to_openai_format(self, entry: Dict) -> Dict:
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": entry['user_message']},
                {"role": "assistant", "content": entry['assistant_response']}
            ]
        }
    
    def convert_to_qwen_format(self, entry: Dict) -> Dict:
        return self.convert_to_openai_format(entry)
    
    def convert_to_sharegpt_format(self, entry: Dict) -> Dict:
        return {
            "conversations": [
                {"from": "system", "value": self.system_message},
                {"from": "human", "value": entry['user_message']},
                {"from": "gpt", "value": entry['assistant_response']}
            ]
        }
    
    def convert_to_alpaca_format(self, entry: Dict) -> Dict:
        return {
            "instruction": f"{self.system_message}\n\n{entry['user_message']}",
            "input": "",
            "output": entry['assistant_response']
        }
    
    def convert_to_chatml_format(self, entry: Dict) -> Dict:
        chatml_text = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{entry['user_message']}<|im_end|>\n"
            f"<|im_start|>assistant\n{entry['assistant_response']}<|im_end|>"
        )
        return {"text": chatml_text}
    
    def convert_to_vicuna_format(self, entry: Dict) -> Dict:
        return {
            "conversations": [
                {"from": "SYSTEM", "value": self.system_message},
                {"from": "USER", "value": entry['user_message']},
                {"from": "ASSISTANT", "value": entry['assistant_response']}
            ]
        }
    
    def convert_to_llama2_format(self, entry: Dict) -> Dict:
        full_instruction = f"<<SYS>>\n{self.system_message}\n<</SYS>>\n\n{entry['user_message']}"
        llama2_text = f"<s>[INST] {full_instruction} [/INST] {entry['assistant_response']} </s>"
        return {"text": llama2_text}
    
    def convert_to_mistral_format(self, entry: Dict) -> Dict:
        combined_instruction = f"{self.system_message}\n\n{entry['user_message']}"
        mistral_text = f"[INST] {combined_instruction} [/INST] {entry['assistant_response']}"
        return {"text": mistral_text}
    
    def convert_entry(self, entry: Dict, format_type: str) -> Optional[Dict]:
        converters = {
            'openai': self.convert_to_openai_format,
            'qwen': self.convert_to_qwen_format,
            'sharegpt': self.convert_to_sharegpt_format,
            'alpaca': self.convert_to_alpaca_format,
            'chatml': self.convert_to_chatml_format,
            'vicuna': self.convert_to_vicuna_format,
            'llama2': self.convert_to_llama2_format,
            'mistral': self.convert_to_mistral_format
        }
        
        converter = converters.get(format_type)
        if not converter:
            logger.error(f"Unknown format type: {format_type}")
            return None
        
        try:
            converted = converter(entry)
            self.stats['converted_entries'] += 1
            return converted
        except Exception as e:
            logger.error(f"Error converting entry: {str(e)}")
            self.stats['failed_entries'] += 1
            return None
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8, 
                   valid_ratio: float = 0.1, test_ratio: float = 0.1, 
                   seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        random.seed(seed)
        
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        total_size = len(shuffled_data)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)

        train_data = shuffled_data[:train_size]
        valid_data = shuffled_data[train_size:train_size + valid_size]
        test_data = shuffled_data[train_size + valid_size:]

        self.stats['train_entries'] = len(train_data)
        self.stats['valid_entries'] = len(valid_data)
        self.stats['test_entries'] = len(test_data)
        
        return train_data, valid_data, test_data
    
    def save_split_data(self, train_data: List[Dict], valid_data: List[Dict], 
                       test_data: List[Dict], format_dir: Path, format_type: str):
        train_path = format_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for entry in train_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(train_data)} training entries to {train_path}")
        
        valid_path = format_dir / "valid.jsonl"
        with open(valid_path, 'w', encoding='utf-8') as f:
            for entry in valid_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(valid_data)} validation entries to {valid_path}")
        
        test_path = format_dir / "test.jsonl"
        with open(test_path, 'w', encoding='utf-8') as f:
            for entry in test_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(test_data)} test entries to {test_path}")
        
        metadata_path = format_dir / "metadata.json"
        metadata = {
            'format': format_type,
            'train_entries': len(train_data),
            'validation_entries': len(valid_data),
            'test_entries': len(test_data),
            'total_entries': len(train_data) + len(valid_data) + len(test_data),
            'conversion_timestamp': datetime.now().isoformat(),
            'source_dataset': 'CyberNative/Code_Vulnerability_Security_DPO',
            'system_message': self.system_message,
            'description': f'Code vulnerability dataset converted to {format_type} chat template format'
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def process_dataset(self, format_type: str = 'qwen', train_ratio: float = 0.8,
                       valid_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        self.stats['format_used'] = format_type
        
        entries = self.load_dataset_entries()
        if not entries:
            logger.error("No entries loaded from dataset")
            return
        
        print(f"\nConverting {len(entries)} entries to {format_type} format...")
        
        format_dir = self.output_dir / format_type
        format_dir.mkdir(parents=True, exist_ok=True)

        converted_entries = []
        for i, entry in enumerate(entries, 1):
            converted = self.convert_entry(entry, format_type)
            if converted:
                converted_entries.append(converted)

            if i % 100 == 0 or i == len(entries):
                percent = (i / len(entries)) * 100
                bar_length = 40
                filled = int(bar_length * i / len(entries))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  Progress: [{bar}] {percent:.1f}% ({i}/{len(entries)})")
        
        if not converted_entries:
            logger.error("No entries were successfully converted")
            return
        
        print(f"\nSplitting data: {train_ratio*100:.0f}% train, {valid_ratio*100:.0f}% valid, {test_ratio*100:.0f}% test")
        train_data, valid_data, test_data = self.split_data(
            converted_entries, train_ratio, valid_ratio, test_ratio, seed
        )
        
        print(f"Saving to {format_dir}/...")
        self.save_split_data(train_data, valid_data, test_data, format_dir, format_type)

        self.display_summary()
    
    def display_summary(self):
        print(f"\nConversion Complete!")
        print("="*50)
        print("Conversion Summary")
        print("="*50)
        print(f"Format Used: {self.stats['format_used']}")
        print(f"Total Entries Loaded: {self.stats['total_entries']}")
        print(f"Successfully Converted: {self.stats['converted_entries']}")
        print(f"Failed Conversions: {self.stats['failed_entries']}")
        print(f"Training Entries: {self.stats['train_entries']}")
        print(f"Validation Entries: {self.stats['valid_entries']}")
        print(f"Test Entries: {self.stats['test_entries']}")
        
        if self.stats['total_entries'] > 0:
            success_rate = (self.stats['converted_entries'] / self.stats['total_entries']) * 100
            print(f"Success Rate: {success_rate:.2f}%")
        
        print("="*50)
        print(f"\nOutput directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Code Vulnerability Security dataset to various chat template formats'
    )
    parser.add_argument(
        '--format',
        choices=['openai', 'qwen', 'sharegpt', 'alpaca', 'chatml', 'vicuna', 'llama2', 'mistral'],
        default='qwen',
        help='Chat template format to convert to (default: qwen)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training set (default: 0.8)'
    )
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=0.1,
        help='Proportion of data for validation set (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Proportion of data for test set (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        default='data',
        help='Output directory for converted files (default: data)'
    )
    parser.add_argument(
        '--all-formats',
        action='store_true',
        help='Convert to all available formats'
    )
    
    args = parser.parse_args()
    
    if args.train_ratio + args.valid_ratio + args.test_ratio > 1.0:
        logger.error("Sum of train, valid, and test ratios must not exceed 1.0")
        return
    
    converter = VulnerabilityDatasetConverter(args.output_dir)

    if args.all_formats:
        formats = ['openai', 'qwen', 'sharegpt', 'alpaca', 'chatml', 'vicuna', 'llama2', 'mistral']
        for format_type in formats:
            print(f"\n{'='*50}")
            print(f"Converting to {format_type} format...")
            print(f"{'='*50}")

            converter.stats = {
                'total_entries': 0,
                'converted_entries': 0,
                'failed_entries': 0,
                'format_used': '',
                'train_entries': 0,
                'valid_entries': 0,
                'test_entries': 0
            }
            
            converter.process_dataset(
                format_type,
                args.train_ratio,
                args.valid_ratio,
                args.test_ratio,
                args.seed
            )
    else:
        converter.process_dataset(
            args.format,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.seed
        )


if __name__ == "__main__":
    main()