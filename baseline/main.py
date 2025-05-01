"""
Main script to demonstrate loading and using the HuggingFaceTransformersModel and DatasetLoader.
"""

import os
from src.model.hf_transformers_model import HuggingFaceTransformersModel
from src.data.dataset_loader import DatasetLoader
from src.data.dataset_splitter import DatasetSplitter
import json

def main():
    # Create an instance of the HuggingFaceTransformersModel
    model = HuggingFaceTransformersModel()
    
    # Load the specified model
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    print(f"Loading model: {model_name}")
    
    try:
        model.load(model_name)
        
        # Print model info
        print("\nModel Information:")
        for key, value in model.model_info.items():
            print(f"  {key}: {value}")
        
        # Load a sample dataset from HuggingFace Hub
        print("\nLoading financial phrasebank dataset...")
        dataset = DatasetLoader.from_huggingface(
            "financial_phrasebank",
            "sentences_allagree"
        )
        
        # Split the dataset
        print("\nSplitting dataset...")
        splits = DatasetSplitter.stratified_split(
            dataset,
            label_column="label",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1
        )
        
        # Print split sizes
        print("\nDataset splits:")
        for split_name, split_dataset in splits.items():
            print(f"  {split_name}: {len(split_dataset)} samples")
        
        # Show samples from training set
        print("\nSample entries from training set:")
        for i, example in enumerate(splits['train'].select(range(3))):
            print(f"\nExample {i+1}:")
            print(f"  Text: {example['text']}")
            print(f"  Label: {example['label']}")
        
        # Make predictions on test set samples
        sample_texts = [example["text"] for example in splits['test'].select(range(3))]
        print("\nMaking predictions on test set samples:")
        predictions = model.predict(sample_texts)
        
        # Display results
        print("\nPrediction Results:")
        for i, result in enumerate(predictions["results"]):
            print(f"\nText {i+1}: \"{result['text']}\"")
            print(f"  Predicted label: {result.get('label', result['prediction'])}")
            print(f"  Probabilities: {[round(p, 4) for p in result['probabilities']]}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 