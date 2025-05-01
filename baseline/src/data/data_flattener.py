from typing import Union, List, Optional
from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, concatenate_datasets, interleave_datasets


class DataFlattener:
    """
    Utility class for flattening datasets by removing split structure.
    Maintains streaming nature for streaming datasets.
    """
    
    @staticmethod
    def flatten(
        data: Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict, List],
        buffer_size: Optional[int] = None
    ) -> Union[Dataset, IterableDataset]:
        """
        Flatten a dataset by removing split structure.
        
        Args:
            data: The data to flatten. Can be:
                - A DatasetDict/IterableDatasetDict (splits will be collapsed)
                - A list of datasets (will be combined)
                - A single dataset (will be returned as is)
            buffer_size: Buffer size for streaming operations
            
        Returns:
            A single dataset without split structure
        """
        # Single dataset case - just return it
        if isinstance(data, (Dataset, IterableDataset)):
            return data
            
        # Get list of datasets to combine
        if isinstance(data, (DatasetDict, IterableDatasetDict)):
            datasets = list(data.values())
        elif isinstance(data, list):
            if not all(isinstance(d, (Dataset, IterableDataset)) for d in data):
                raise ValueError("All items in list must be datasets")
            datasets = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # Handle streaming vs regular datasets
        is_streaming = any(isinstance(d, IterableDataset) for d in datasets)
        
        if is_streaming:
            # Convert any regular datasets to streaming if mixed
            streaming_datasets = [
                d if isinstance(d, IterableDataset) else d.to_iterable_dataset(buffer_size=buffer_size)
                for d in datasets
            ]
            # For streaming, use interleave_datasets with sequential processing
            return interleave_datasets(
                streaming_datasets,
                probabilities=None,  # Equal probability
                stopping_strategy='all_exhausted',
                buffer_size=buffer_size
            )
        else:
            # For regular datasets, use concatenate_datasets
            return concatenate_datasets(datasets)
    
    @staticmethod
    def flatten_with_source(
        data: Union[DatasetDict, IterableDatasetDict],
        source_column: str = "source",
        buffer_size: Optional[int] = None
    ) -> Union[Dataset, IterableDataset]:
        """
        Flatten a dataset dictionary while preserving source information.
        
        Args:
            data: The dataset dictionary to flatten
            source_column: Name of the column to store source information
            buffer_size: Buffer size for streaming operations
            
        Returns:
            A single flattened dataset with source information
        """
        if not isinstance(data, (DatasetDict, IterableDatasetDict)):
            raise ValueError("Data must be a DatasetDict or IterableDatasetDict")
        
        is_streaming = isinstance(data, IterableDatasetDict)
        datasets = []
        
        for split_name, dataset in data.items():
            if is_streaming:
                # For streaming, wrap the dataset with a source-adding generator
                def add_source(iterable, source):
                    for example in iterable:
                        example[source_column] = source
                        yield example
                
                labeled_dataset = IterableDataset.from_generator(
                    lambda: add_source(dataset, split_name)
                )
            else:
                # For regular datasets, add column directly
                labeled_dataset = dataset.add_column(
                    source_column,
                    [split_name] * len(dataset)
                )
            
            datasets.append(labeled_dataset)
        
        # Use the main flatten method to combine datasets
        return DataFlattener.flatten(datasets, buffer_size=buffer_size) 