"""Custom pipeline for generating fine-tuning data.

This module provides filtering and export utilities for creating high-quality
training datasets from Themis generation results.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from themis.core import entities

logger = logging.getLogger(__name__)


class FinetuningDataFilter:
    """Filter and format generation results for fine-tuning.
    
    This class filters generation records based on various criteria and exports
    them in formats suitable for fine-tuning (JSONL, CSV, etc.).
    
    Args:
        only_correct: If True, only include records with correct answers (default: True).
        min_length: Minimum response length in characters (default: None).
        max_length: Maximum response length in characters (default: None).
        custom_filter: Optional custom filter function taking a record and returning bool.
    
    Example:
        >>> filter = FinetuningDataFilter(only_correct=True, min_length=10)
        >>> filtered = filter.filter_records(generation_records)
        >>> filter.export_jsonl(filtered, "training_data.jsonl")
    """
    
    def __init__(
        self,
        only_correct: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        custom_filter: Optional[Callable[[dict], bool]] = None,
    ):
        self.only_correct = only_correct
        self.min_length = min_length
        self.max_length = max_length
        self.custom_filter = custom_filter
        
    def filter_records(
        self, 
        records: list[entities.GenerationRecord],
        evaluations: Optional[list[entities.EvaluationRecord]] = None
    ) -> list[dict]:
        """Filter and format records for fine-tuning.
        
        Args:
            records: List of generation records to filter.
            evaluations: Optional list of evaluation records for quality filtering.
        
        Returns:
            List of filtered records in dictionary format.
        
        Note:
            Records are filtered based on:
            1. No generation errors
            2. Valid output exists
            3. Correctness (if only_correct=True and evaluations provided)
            4. Length constraints (if min_length or max_length set)
            5. Custom filter (if provided)
        """
        filtered = []
        skipped_counts = {
            "error": 0,
            "no_output": 0,
            "incorrect": 0,
            "too_short": 0,
            "too_long": 0,
            "custom_filter": 0
        }
        
        for i, record in enumerate(records):
            # Skip failed generations
            if record.error is not None:
                skipped_counts["error"] += 1
                logger.debug(f"Skipping record {i}: has error")
                continue
                
            if not record.output:
                skipped_counts["no_output"] += 1
                logger.debug(f"Skipping record {i}: no output")
                continue
            
            # Check correctness if evaluations provided
            if self.only_correct and evaluations:
                if i < len(evaluations):
                    eval_record = evaluations[i]
                    # Check if any metric shows success
                    is_correct = any(
                        score.value > 0.5 
                        for score in eval_record.scores
                    )
                    if not is_correct:
                        skipped_counts["incorrect"] += 1
                        logger.debug(f"Skipping record {i}: incorrect")
                        continue
            
            # Format for fine-tuning
            prompt_text = (
                record.task.prompt.text 
                if hasattr(record.task.prompt, 'text') 
                else str(record.task.prompt)
            )
            response_text = record.output.text
            
            # Apply length constraints
            if self.min_length is not None and len(response_text) < self.min_length:
                skipped_counts["too_short"] += 1
                logger.debug(f"Skipping record {i}: too short ({len(response_text)} < {self.min_length})")
                continue
            
            if self.max_length is not None and len(response_text) > self.max_length:
                skipped_counts["too_long"] += 1
                logger.debug(f"Skipping record {i}: too long ({len(response_text)} > {self.max_length})")
                continue
            
            formatted = {
                "prompt": prompt_text,
                "completion": response_text,
                "metadata": {
                    "sample_id": record.task.metadata.get("dataset_id"),
                    "model": record.task.model.identifier,
                    "temperature": record.task.sampling.temperature,
                    "prompt_length": len(prompt_text),
                    "completion_length": len(response_text),
                }
            }
            
            # Apply custom filter
            if self.custom_filter is not None:
                if not self.custom_filter(formatted):
                    skipped_counts["custom_filter"] += 1
                    logger.debug(f"Skipping record {i}: custom filter")
                    continue
            
            filtered.append(formatted)
        
        # Log filtering statistics
        logger.info(
            f"Filtering complete: {len(filtered)}/{len(records)} records kept. "
            f"Skipped: {skipped_counts}"
        )
        
        return filtered
    
    def export_jsonl(self, records: list[dict], output_path: str) -> None:
        """Export filtered records to JSONL file.
        
        Args:
            records: Filtered records to export.
            output_path: Path to output JSONL file.
        
        Note:
            Each line in the output file is a JSON object suitable for fine-tuning.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(records)} records to {output_path}")
            print(f"Exported {len(records)} records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export JSONL: {e}")
            raise
    
    def export_csv(self, records: list[dict], output_path: str) -> None:
        """Export filtered records to CSV file.
        
        Args:
            records: Filtered records to export.
            output_path: Path to output CSV file.
        
        Note:
            Metadata is stored as a JSON string in the CSV.
        """
        if not records:
            logger.warning("No records to export")
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['prompt', 'completion', 'metadata']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in records:
                    # Convert metadata to JSON string for CSV
                    row = {
                        'prompt': record['prompt'],
                        'completion': record['completion'],
                        'metadata': json.dumps(record.get('metadata', {}))
                    }
                    writer.writerow(row)
            
            logger.info(f"Exported {len(records)} records to {output_path}")
            print(f"Exported {len(records)} records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise


__all__ = ["FinetuningDataFilter"]
