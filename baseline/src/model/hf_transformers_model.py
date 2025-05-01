from typing import Any, Dict, List, Optional, Union
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from .model import Model


class HuggingFaceTransformersModel(Model):
    """
    Implementation of the Model interface for Hugging Face Transformer models.
    
    This class provides functionality to work with pre-trained transformer models
    from the Hugging Face model hub.
    """
    
    def __init__(self):
        """Initialize the HuggingFaceTransformersModel."""
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load a pre-trained Hugging Face transformer model.
        
        Args:
            model_path: Path or identifier of the model on the Hugging Face model hub
                        (e.g., 'cardiffnlp/twitter-roberta-base-sentiment-latest')
            **kwargs: Additional parameters for model loading
                - task_type: Type of task (e.g., 'sequence-classification')
                - revision: Model revision to use
                - token: Hugging Face token for accessing gated models (replaces deprecated use_auth_token)
        """
        try:
            # Store the model name for reference
            self.model_name = model_path
            
            # Extract kwargs
            revision = kwargs.get('revision', 'main')
            token = kwargs.get('token', False)
            
            # Load the model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                revision=revision,
                token=token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                token=token
            )
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Successfully loaded model: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Make predictions using the model.
        
        Args:
            inputs: Input text or list of texts for prediction
            
        Returns:
            Dictionary containing prediction results with labels and scores
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions")
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]
        
        try:
            # Tokenize the inputs
            encoded_inputs = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Get the logits
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the predicted class (highest probability)
            predictions = torch.argmax(probs, dim=1)
            
            # Convert to numpy for easier handling
            predictions = predictions.cpu().numpy()
            probs = probs.cpu().numpy()
            
            # Get the label mapping if available
            labels = getattr(self.model.config, 'id2label', None)
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probs)):
                result = {
                    "text": inputs[i],
                    "prediction": int(pred),
                    "probabilities": prob.tolist()
                }
                
                # Add label if available
                if labels:
                    result["label"] = labels[pred]
                
                results.append(result)
            
            return {
                "results": results,
                "model_name": self.model_name
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def save(self, save_path: str, **kwargs) -> None:
        """
        Save the model to a given path.
        
        Args:
            save_path: Path where the model should be saved
            **kwargs: Additional model-specific saving parameters
        """
        # This will be implemented later
        pass
    
    def evaluate(self, test_data: Any, metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Data to evaluate the model on
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and their values
        """
        # This will be implemented later
        pass
    
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
        # This will be implemented later
        pass
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information (e.g., type, size, etc.)
        """
        # This will be implemented later
        return {
            "model_name": self.model_name,
            "model_type": "HuggingFace Transformer",
            "device": self.device
        } 