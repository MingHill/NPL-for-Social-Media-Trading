from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Utility class for splitting datasets into train/validation/test sets.
    Supports various splitting strategies including:
    - Random splitting
    - Stratified splitting (maintaining label distribution)
    - Time-based splitting
    """
    
    @staticmethod
    def random_split(
        dataset: Dataset,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Randomly split a dataset into train/validation/test sets.
        
        Args:
            dataset: The dataset to split
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        splits = {}
        remaining_data = dataset
        
        # Split test set if specified
        if test_size:
            remaining_data, test_data = remaining_data.train_test_split(
                test_size=test_size/(test_size + train_size + (val_size or 0)),
                seed=seed
            ).values()
            splits['test'] = test_data
            
        # Split validation set if specified
        if val_size:
            remaining_data, val_data = remaining_data.train_test_split(
                test_size=val_size/(val_size + train_size),
                seed=seed
            ).values()
            splits['validation'] = val_data
            
        splits['train'] = remaining_data
        
        return DatasetDict(splits)
    
    @staticmethod
    def stratified_split(
        dataset: Dataset,
        label_column: str,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Split a dataset while maintaining the same label distribution across splits.
        
        Args:
            dataset: The dataset to split
            label_column: Name of the column containing labels
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        splits = {}
        remaining_data = dataset
        
        # Split test set if specified
        if test_size:
            remaining_data, test_data = remaining_data.train_test_split(
                test_size=test_size/(test_size + train_size + (val_size or 0)),
                seed=seed,
                stratify_by_column=label_column
            ).values()
            splits['test'] = test_data
            
        # Split validation set if specified
        if val_size:
            remaining_data, val_data = remaining_data.train_test_split(
                test_size=val_size/(val_size + train_size),
                seed=seed,
                stratify_by_column=label_column
            ).values()
            splits['validation'] = val_data
            
        splits['train'] = remaining_data
        
        return DatasetDict(splits)
    
    @staticmethod
    def temporal_split(
        dataset: Dataset,
        timestamp_column: str,
        train_size: float = 0.8,
        val_size: Optional[float] = 0.1,
        test_size: Optional[float] = 0.1
    ) -> DatasetDict:
        """
        Split a dataset based on temporal ordering.
        
        Args:
            dataset: The dataset to split
            timestamp_column: Name of the column containing timestamps
            train_size: Proportion of data for training (default: 0.8)
            val_size: Proportion of data for validation (default: 0.1)
            test_size: Proportion of data for testing (default: 0.1)
            
        Returns:
            DatasetDict containing the splits
        """
        if not np.isclose(train_size + (val_size or 0) + (test_size or 0), 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        # Sort dataset by timestamp
        sorted_dataset = dataset.sort(timestamp_column)
        
        total_size = len(sorted_dataset)
        splits = {}
        
        # Calculate split indices
        train_end = int(total_size * train_size)
        val_end = train_end + int(total_size * (val_size or 0)) if val_size else train_end
        
        # Create splits
        splits['train'] = sorted_dataset.select(range(train_end))
        
        if val_size:
            splits['validation'] = sorted_dataset.select(range(train_end, val_end))
            
        if test_size:
            splits['test'] = sorted_dataset.select(range(val_end, total_size))
            
        return DatasetDict(splits)
    
    @staticmethod
    def custom_split(
        dataset: Dataset,
        split_fn: callable,
        **kwargs
    ) -> DatasetDict:
        """
        Split a dataset using a custom splitting function.
        
        Args:
            dataset: The dataset to split
            split_fn: Custom function that takes a dataset and returns a DatasetDict
            **kwargs: Additional arguments passed to split_fn
            
        Returns:
            DatasetDict containing the splits
        """
        return split_fn(dataset, **kwargs) 