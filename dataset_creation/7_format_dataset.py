#!/usr/bin/env python3

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetFormatter:
    
    def __init__(self):
        self.patterns = {
            'header': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'bullet': re.compile(r'^(\s*[-*+])\s+(.+)$', re.MULTILINE),
            'numbered': re.compile(r'^(\s*\d+\.)\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'inline_code': re.compile(r'`[^`]+`'),
            'bold': re.compile(r'\*\*[^*]+\*\*'),
            'section_separator': re.compile(r'^---+$', re.MULTILINE),
            'definition': re.compile(r'^(\w+[^:]*:)\s*(.+)$', re.MULTILINE)
        }
    
    def format_content(self, content: str) -> str:
        if not content:
            return content
        
        # First, fix the main issue: content is all on one line
        # Add newlines before common markdown patterns
        content = re.sub(r'(\S)\s*(---+)\s*(\S)', r'\1\n\n\2\n\n\3', content)  # Section separators
        content = re.sub(r'(\S)\s+(#{1,6}\s+)', r'\1\n\n\2', content)  # Headers
        content = re.sub(r'(\.)\s*(-\s+\w)', r'\1\n\n\2', content)  # Bullet lists after sentences
        content = re.sub(r'(\*\*[^*]+\*\*:)', r'\n\n\1', content)  # Bold definitions
        
        # Preserve code blocks first
        code_blocks = []
        def preserve_code(match):
            code_blocks.append(match.group())
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        formatted = re.sub(self.patterns['code_block'], preserve_code, content)

        formatted = self._format_headers(formatted)
        formatted = self._format_lists(formatted)
        formatted = self._format_sections(formatted)
        formatted = self._format_definitions(formatted)
        formatted = self._clean_spacing(formatted)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            formatted = formatted.replace(f"__CODE_BLOCK_{i}__", block)
        
        return formatted
    
    def _format_headers(self, text: str) -> str:
        # First, fix inline headers that aren't on their own lines
        # Replace patterns like "text ### Header" with "text\n\n### Header"
        text = re.sub(r'([^\n#])(\s*)(#{1,6}\s+\*{0,2}[\w\s]+\*{0,2})', r'\1\n\n\3', text)

        lines = text.split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            if self.patterns['header'].match(line.strip()):
                # Add blank line before header (if not at start and previous line isn't blank)
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
                # Add blank line after header (if next line exists and isn't blank)
                if i < len(lines) - 1 and lines[i + 1].strip():
                    formatted_lines.append('')
            else:
                # Skip adding the line if it would be a duplicate blank from header formatting
                if not (line.strip() == '' and formatted_lines and formatted_lines[-1] == ''):
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_lists(self, text: str) -> str:
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        
        for i, line in enumerate(lines):
            is_bullet = bool(self.patterns['bullet'].match(line))
            is_numbered = bool(self.patterns['numbered'].match(line))
            is_list_item = is_bullet or is_numbered
            
            if is_list_item:
                # Starting a new list - add blank line before if needed
                if not in_list and i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                in_list = True
                formatted_lines.append(line)
            else:
                # Ending a list - add blank line after if needed
                if in_list and line.strip():
                    formatted_lines.append('')
                in_list = False
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_sections(self, text: str) -> str:
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if self.patterns['section_separator'].match(line.strip()):
                # Add blank lines around separator
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
                if i < len(lines) - 1:
                    formatted_lines.append('')
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_definitions(self, text: str) -> str:
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            # Check for patterns like "**Term:**" or "Term:"
            if ':' in line and (line.strip().startswith('**') or 
                               (i > 0 and lines[i-1].strip() == '') or
                               self.patterns['definition'].match(line.strip())):
                # This looks like a definition - ensure it has proper spacing
                if i > 0 and formatted_lines and formatted_lines[-1].strip() and not formatted_lines[-1].startswith('#'):
                    formatted_lines.append('')
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _clean_spacing(self, text: str) -> str:
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove excessive blank lines (max 2 consecutive)
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)

        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def process_message(self, message: Dict) -> Dict:
        if 'content' not in message:
            return message
        
        formatted_message = message.copy()

        if message.get('role') == 'assistant':
            # Assistant responses need the most formatting
            formatted_message['content'] = self.format_content(message['content'])
        elif message.get('role') == 'user':
            # User messages usually need less formatting
            formatted_message['content'] = self._clean_spacing(message['content'])
        # System messages typically stay as-is
        
        return formatted_message
    
    def process_jsonl_entry(self, entry: Dict) -> Dict:
        formatted_entry = entry.copy()

        if 'messages' in entry:
            formatted_entry['messages'] = [
                self.process_message(msg) for msg in entry['messages']
            ]

        elif 'instruction' in entry and 'response' in entry:
            if 'instruction' in entry:
                formatted_entry['instruction'] = self._clean_spacing(entry['instruction'])
            if 'response' in entry:
                formatted_entry['response'] = self.format_content(entry['response'])

        elif 'text' in entry:
            formatted_entry['text'] = self.format_content(entry['text'])

        formatted_entry['_reformatted'] = True
        formatted_entry['_reformatted_at'] = datetime.now().isoformat()
        
        return formatted_entry
    
    def process_file(self, input_path: str, output_path: str, limit: Optional[int] = None):
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {input_path} -> {output_path}")
        
        processed = 0
        errors = 0
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Count total lines for progress bar
            total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
            if limit:
                total_lines = min(total_lines, limit)

            infile.seek(0)
            
            for line in tqdm(infile, total=total_lines, desc="Formatting"):
                if limit and processed >= limit:
                    break
                
                try:
                    entry = json.loads(line.strip())
                    formatted_entry = self.process_jsonl_entry(entry)
                    outfile.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing entry: {e}")
                    errors += 1
                    # Write original entry on error
                    outfile.write(line)
        
        logger.info(f"Processing complete: {processed} entries formatted, {errors} errors")

        self.generate_summary(output_path)
    
    def generate_summary(self, output_path: str):
        output_file = Path(output_path)
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"
        
        stats = {
            'file': output_file.name,
            'timestamp': datetime.now().isoformat(),
            'total_entries': 0,
            'avg_message_length': 0,
            'total_characters': 0,
            'formatting_changes': 0
        }
        
        lengths = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                stats['total_entries'] += 1
                entry = json.loads(line)

                if 'messages' in entry:
                    for msg in entry['messages']:
                        if 'content' in msg:
                            length = len(msg['content'])
                            lengths.append(length)
                            stats['total_characters'] += length

                if entry.get('_reformatted'):
                    stats['formatting_changes'] += 1
        
        if lengths:
            stats['avg_message_length'] = sum(lengths) / len(lengths)
        
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Format JSONL dataset with proper markdown structure using regex')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of entries to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    formatter = DatasetFormatter()

    try:
        formatter.process_file(args.input, args.output, args.limit)
        logger.info(f"Formatting complete! Output saved to {args.output}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == '__main__':
    main()