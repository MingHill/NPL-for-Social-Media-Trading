from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset


class DatasetLoader:
    """
    Utility class for loading data from various sources into a HuggingFace Dataset object.
    Supports loading from:
    - CSV files
    - JSON files
    - Pandas DataFrames
    - HuggingFace Hub
    - Text files
    """
    
    @staticmethod
    def from_csv(
        path: Union[str, Path],
        text_column: str,
        label_column: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from a CSV file.
        
        Args:
            path: Path to the CSV file
            text_column: Name of the column containing the text data
            label_column: Name of the column containing the labels (optional)
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            HuggingFace Dataset object
        """
        df = pd.read_csv(path, **kwargs)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def from_json(
        path: Union[str, Path],
        text_field: str,
        label_field: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from a JSON file.
        
        Args:
            path: Path to the JSON file
            text_field: Field name containing the text data
            label_field: Field name containing the labels (optional)
            **kwargs: Additional arguments passed to pd.read_json
            
        Returns:
            HuggingFace Dataset object
        """
        df = pd.read_json(path, **kwargs)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def from_pandas(df: pd.DataFrame) -> Dataset:
        """
        Load a dataset from a pandas DataFrame.
        
        Args:
            df: Pandas DataFrame containing the data
            
        Returns:
            HuggingFace Dataset object
        """
        return Dataset.from_pandas(df)
    
    @staticmethod
    def from_huggingface(
        dataset_name: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from the HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            subset: Name of the dataset subset/configuration (optional)
            split: Which split of the data to load (optional)
            **kwargs: Additional arguments passed to load_dataset
            
        Returns:
            HuggingFace Dataset object
        """
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, **kwargs)
        else:
            dataset = load_dataset(dataset_name, split=split, **kwargs)
        return dataset
    
    @staticmethod
    def from_text(
        path: Union[str, Path],
        text_files_or_patterns: Union[str, List[str]],
        labels: Optional[List[Union[str, int]]] = None,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from text files.
        
        Args:
            path: Directory containing the text files
            text_files_or_patterns: List of file names or glob patterns
            labels: Optional list of labels corresponding to the files
            **kwargs: Additional arguments passed to Dataset.from_text
            
        Returns:
            HuggingFace Dataset object
        """
        return Dataset.from_text(
            path,
            files=text_files_or_patterns,
            labels=labels,
            **kwargs
        ) 