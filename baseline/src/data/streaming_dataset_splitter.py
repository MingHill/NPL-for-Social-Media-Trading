from typing import Dict, Iterator, Optional
from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict
import numpy as np
import random


class StreamingDatasetSplitter:
    """
    Utility class for splitting streaming/iterable datasets into train/validation/test sets.
    Uses reservoir sampling and on-the-fly decisions for streaming-compatible splitting.
    """
    
    @staticmethod
    def random_split(
        dataset: IterableDataset,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1,
        seed: int = 42,
        buffer_size: int = 1000
    ) -> IterableDatasetDict:
        """
        Randomly split a streaming dataset using on-the-fly decisions.
        
        Args:
            dataset: The streaming dataset to split
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            seed: Random seed for reproducibility
            buffer_size: Size of the shuffle buffer for randomization
            
        Returns:
            IterableDatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        random.seed(seed)
        
        def split_generator(dataset: IterableDataset):
            for example in dataset:
                # Generate random number to decide split
                split_decision = random.random()
                
                if test_size and split_decision < test_size:
                    yield ('test', example)
                elif val_size and split_decision < (test_size + val_size):
                    yield ('validation', example)
                else:
                    yield ('train', example)
        
        # Create split iterators
        split_streams = {
            'train': dataset.filter(
                lambda x, idx: random.random() >= (test_size or 0) + (val_size or 0),
                with_indices=True
            ).shuffle(buffer_size=buffer_size, seed=seed),
        }
        
        if val_size:
            split_streams['validation'] = dataset.filter(
                lambda x, idx: (test_size or 0) <= random.random() < (test_size or 0) + val_size,
                with_indices=True
            ).shuffle(buffer_size=buffer_size, seed=seed)
            
        if test_size:
            split_streams['test'] = dataset.filter(
                lambda x, idx: random.random() < test_size,
                with_indices=True
            ).shuffle(buffer_size=buffer_size, seed=seed)
            
        return IterableDatasetDict(split_streams)
    
    @staticmethod
    def stratified_split(
        dataset: IterableDataset,
        label_column: str,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1,
        seed: int = 42,
        buffer_size: int = 1000
    ) -> IterableDatasetDict:
        """
        Split a streaming dataset while approximately maintaining label distribution.
        Uses reservoir sampling to maintain proportions.
        
        Args:
            dataset: The streaming dataset to split
            label_column: Name of the column containing labels
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            seed: Random seed for reproducibility
            buffer_size: Size of the shuffle buffer for randomization
            
        Returns:
            IterableDatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        random.seed(seed)
        
        # Track label counts for each split to maintain distribution
        label_counts = {}
        
        def should_accept_for_split(label, split):
            """Decide whether to accept an example for a split based on current proportions"""
            if label not in label_counts:
                label_counts[label] = {'train': 0, 'validation': 0, 'test': 0, 'total': 0}
            
            counts = label_counts[label]
            target_prop = {'train': train_size, 'validation': val_size, 'test': test_size}[split]
            current_prop = counts[split] / (counts['total'] + 1) if counts['total'] > 0 else 0
            
            return current_prop < target_prop
        
        def split_generator(dataset: IterableDataset):
            for example in dataset:
                label = example[label_column]
                available_splits = []
                
                if test_size and should_accept_for_split(label, 'test'):
                    available_splits.append('test')
                if val_size and should_accept_for_split(label, 'validation'):
                    available_splits.append('validation')
                if should_accept_for_split(label, 'train'):
                    available_splits.append('train')
                
                if not available_splits:
                    chosen_split = 'train'  # Default to train if no split needs more examples
                else:
                    chosen_split = random.choice(available_splits)
                
                # Update counts
                if label not in label_counts:
                    label_counts[label] = {'train': 0, 'validation': 0, 'test': 0, 'total': 0}
                label_counts[label][chosen_split] += 1
                label_counts[label]['total'] += 1
                
                yield (chosen_split, example)
        
        # Create split streams
        splits = {}
        for example in split_generator(dataset):
            split_name, data = example
            if split_name not in splits:
                splits[split_name] = []
            splits[split_name].append(data)
        
        # Convert to IterableDataset
        split_streams = {
            name: Dataset.from_list(examples).to_iterable_dataset(buffer_size=buffer_size)
            for name, examples in splits.items()
        }
        
        return IterableDatasetDict(split_streams)
    
    @staticmethod
    def temporal_split(
        dataset: IterableDataset,
        timestamp_column: str,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1,
        buffer_size: int = 1000
    ) -> IterableDatasetDict:
        """
        Split a streaming dataset based on temporal ordering.
        Assumes data is already sorted by timestamp.
        
        Args:
            dataset: The streaming dataset to split
            timestamp_column: Name of the column containing timestamps
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            buffer_size: Size of the buffer for windowing
            
        Returns:
            IterableDatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
        
        def get_split_for_position(position: float) -> str:
            """Determine split based on relative position in stream"""
            if position < train_size:
                return 'train'
            elif position < train_size + (val_size or 0):
                return 'validation'
            else:
                return 'test'
        
        # Create windowed buffer to track progress
        window_buffer = []
        total_seen = 0
        
        def split_generator(dataset: IterableDataset):
            nonlocal total_seen
            
            for example in dataset:
                window_buffer.append(example)
                if len(window_buffer) >= buffer_size:
                    # Process and yield from buffer
                    for buffered_example in window_buffer:
                        position = total_seen / (total_seen + 1)
                        split = get_split_for_position(position)
                        total_seen += 1
                        yield (split, buffered_example)
                    window_buffer.clear()
        
        # Create split streams
        splits = {}
        for example in split_generator(dataset):
            split_name, data = example
            if split_name not in splits:
                splits[split_name] = []
            splits[split_name].append(data)
        
        # Convert to IterableDataset
        split_streams = {
            name: Dataset.from_list(examples).to_iterable_dataset(buffer_size=buffer_size)
            for name, examples in splits.items()
        }
        
        return IterableDatasetDict(split_streams) 