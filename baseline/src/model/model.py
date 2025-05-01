from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class Model(ABC):
    """
    Abstract base class that defines the interface for all models.
    
    This class serves as a contract that all model implementations
    (e.g., HuggingFaceModel, TorchModel, etc.) must fulfill by
    implementing all the abstract methods defined here.
    """
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load a model from a given path.
        
        Args:
            model_path: Path to the model file or directory
            **kwargs: Additional model-specific loading parameters
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Union[List, np.ndarray, Dict[str, Any]]) -> Any:
        """
        Make predictions using the model.
        
        Args:
            inputs: Input data for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str, **kwargs) -> None:
        """
        Save the model to a given path.
        
        Args:
            save_path: Path where the model should be saved
            **kwargs: Additional model-specific saving parameters
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any, metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Data to evaluate the model on
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and their values
        """
        pass
    
    @abstractmethod
    def fine_tune(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs) -> Dict[str, List[float]]:
        """
        Fine-tune the model on training data.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history (e.g., loss values)
        """
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information (e.g., type, size, etc.)
        """
        pass
