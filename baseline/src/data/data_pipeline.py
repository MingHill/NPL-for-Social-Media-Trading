from typing import Optional, Union, Dict, Any
from enum import Enum, auto
from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict
from .dataset_loader import DatasetLoader
from .dataset_splitter import DatasetSplitter
from .streaming_dataset_splitter import StreamingDatasetSplitter
from .data_flattener import DataFlattener


class LoadStrategy(Enum):
    """Enumeration of available dataset loading strategies"""
    REGULAR = auto()  # Load entire dataset into memory
    STREAMING = auto()  # Load dataset in streaming mode
    REGULAR_SPLIT = auto()  # Dataset that comes with splits, load in memory
    STREAMING_SPLIT = auto()  # Dataset that comes with splits, load streaming


class SplitStrategy(Enum):
    """Enumeration of available dataset splitting strategies"""
    NONE = auto()  # No splitting needed (e.g., already split)
    RANDOM = auto()  # Random splitting
    STRATIFIED = auto()  # Split maintaining label distribution
    TEMPORAL = auto()  # Split based on temporal ordering


class DataPipeline:
    """
    A configurable data pipeline that handles dataset loading and splitting.
    
    Attributes:
        name: Name or path of the dataset
        load_strategy: Strategy to use for loading the dataset
        split_strategy: Strategy to use for splitting the dataset (if needed)
    """
    
    def __init__(
        self,
        name: str,
        load_strategy: LoadStrategy,
        split_strategy: SplitStrategy = SplitStrategy.NONE,
        streaming_buffer_size: int = 1000,
        should_flatten: bool = False,
        preserve_split_source: bool = False,
        **kwargs
    ):
        """
        Initialize the data pipeline.
        
        Args:
            name: Name or path of the dataset
            load_strategy: Strategy to use for loading the dataset
            split_strategy: Strategy to use for splitting (default: NONE)
            streaming_buffer_size: Buffer size for streaming operations
            should_flatten: Whether to flatten the dataset before splitting
            preserve_split_source: Whether to preserve source split information when flattening
            **kwargs: Additional arguments for loading/splitting
                - split_sizes: Dict with 'train', 'validation', 'test' sizes
                - label_column: Column name for stratified splitting
                - timestamp_column: Column name for temporal splitting
                - seed: Random seed for reproducibility
                - subset: Dataset subset/configuration name
                - split: Which split to load (for pre-split datasets)
                - source_column: Name of column for storing source split info
        """
        self.name = name
        self.load_strategy = load_strategy
        self.split_strategy = split_strategy
        self.streaming_buffer_size = streaming_buffer_size
        self.should_flatten = should_flatten
        self.preserve_split_source = preserve_split_source
        self.kwargs = kwargs
        
        # Set default split sizes if not provided
        self.split_sizes = kwargs.get('split_sizes', {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        })
    
    def load(self) -> Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict]:
        """
        Load the dataset according to the specified strategy.
        
        Returns:
            The loaded dataset
        """
        # Extract dataset loading specific arguments
        load_args = {
            'dataset_name': self.name,
            'subset': self.kwargs.get('subset'),
            'split': self.kwargs.get('split')
        }
        
        # Filter out known pipeline-specific arguments and streaming config
        dataset_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k not in {
                'split_sizes', 'label_column', 'timestamp_column',
                'seed', 'subset', 'split', 'source_column',
                'buffer_size', 'streaming_buffer_size'
            }
        }
        
        # Add dataset-specific kwargs
        load_args.update(dataset_kwargs)
        
        # Prepare streaming configuration if needed
        streaming_config = None
        if self.load_strategy in [LoadStrategy.STREAMING, LoadStrategy.STREAMING_SPLIT]:
            streaming_config = {'buffer_size': self.streaming_buffer_size}
        
        # Load dataset using appropriate strategy
        if self.load_strategy == LoadStrategy.REGULAR:
            dataset = DatasetLoader.from_huggingface(**load_args)
        elif self.load_strategy == LoadStrategy.STREAMING:
            dataset = DatasetLoader.from_huggingface(
                streaming=streaming_config,
                **load_args
            )
        elif self.load_strategy == LoadStrategy.REGULAR_SPLIT:
            dataset = DatasetLoader.from_huggingface(**load_args)
        elif self.load_strategy == LoadStrategy.STREAMING_SPLIT:
            dataset = DatasetLoader.from_huggingface(
                streaming=streaming_config,
                **load_args
            )
        else:
            raise ValueError(f"Unsupported load strategy: {self.load_strategy}")
        
        # Flatten if requested
        if self.should_flatten:
            if self.preserve_split_source and isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                dataset = DataFlattener.flatten_with_source(
                    dataset,
                    source_column=self.kwargs.get('source_column', 'source'),
                    buffer_size=self.streaming_buffer_size
                )
            else:
                dataset = DataFlattener.flatten(
                    dataset,
                    buffer_size=self.streaming_buffer_size
                )
        
        return dataset
    
    def split(
        self,
        dataset: Union[Dataset, IterableDataset]
    ) -> Union[DatasetDict, IterableDatasetDict]:
        """
        Split the dataset according to the specified strategy.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Split dataset
        """
        if self.split_strategy == SplitStrategy.NONE:
            if isinstance(dataset, (Dataset, IterableDataset)):
                return DatasetDict({'train': dataset})
            return dataset
        
        # Select appropriate splitter based on dataset type
        is_streaming = isinstance(dataset, IterableDataset)
        splitter = StreamingDatasetSplitter if is_streaming else DatasetSplitter
        
        # Prepare common arguments
        split_args = {
            'train_size': self.split_sizes['train'],
            'val_size': self.split_sizes.get('validation'),
            'test_size': self.split_sizes.get('test'),
            'seed': self.kwargs.get('seed', 42)
        }
        
        # Add buffer_size only for streaming datasets
        if is_streaming:
            split_args['buffer_size'] = self.streaming_buffer_size
        
        # Handle different splitting strategies
        if self.split_strategy == SplitStrategy.RANDOM:
            return splitter.random_split(dataset, **split_args)
        
        elif self.split_strategy == SplitStrategy.STRATIFIED:
            if 'label_column' not in self.kwargs:
                raise ValueError("label_column must be specified for stratified splitting")
            
            split_args['label_column'] = self.kwargs['label_column']
            return splitter.stratified_split(dataset, **split_args)
        
        elif self.split_strategy == SplitStrategy.TEMPORAL:
            if 'timestamp_column' not in self.kwargs:
                raise ValueError("timestamp_column must be specified for temporal splitting")
            
            split_args['timestamp_column'] = self.kwargs['timestamp_column']
            return splitter.temporal_split(dataset, **split_args)
        
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def prepare(self) -> Union[DatasetDict, IterableDatasetDict]:
        """
        Execute the full data pipeline: load, flatten (if requested), and split the dataset.
        
        Returns:
            Processed dataset
        """
        dataset = self.load()
        
        if self.split_strategy != SplitStrategy.NONE:
            dataset = self.split(dataset)
            
        return dataset 