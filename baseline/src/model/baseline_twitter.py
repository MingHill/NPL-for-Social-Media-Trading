from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import torch 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix 


def analyze_sentiment(text, tokenizer, model):
    '''
        Processes a single tweet and predics is it is Negative(0), Neutral(1) or Positive(2)
    '''
    # Preprocess text (replace usernames and links with placeholders)
    text = ' '.join(['@user' if word.startswith('@') else 'http' if word.startswith('http') else word for word in text.split()])
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        sentiment = torch.argmax(probabilities, dim=-1).item()
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        return sentiment_labels[sentiment], probabilities[0][sentiment].item()
    
def predict(df, tokenizer, model):
    correct = 0 
    total = 0 
    i = 1

    positive_correct, positive_total, negative_correct, negative_total, neutral_correct, neutral_total = 0,0,0,0,0,0
    
    y_true = []
    y_pred = []

    for index, row in df.iterrows():
        text = row['tweet']
        actual_label = row['sentiment']
        
        predicted_sentiment, confidence = analyze_sentiment(text, tokenizer, model)
        sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        predicted_label = sentiment_mapping[predicted_sentiment]

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        print(f"{i}: Predicted Label: {predicted_label} | Actual Label:{actual_label}")
        if predicted_label == actual_label: 
                correct += 1 
        if predicted_label == 0: 
            # negative counts 
            negative_total += 1 
            if predicted_label == actual_label: 
                negative_correct += 1
        elif predicted_label == 1: 
            # neutral counts
            neutral_total += 1 
            if predicted_label == actual_label: 
                neutral_correct += 1
        else: 
            # positive cunts
            positive_total += 1
            if predicted_label == actual_label: 
                positive_correct += 1

        total += 1 
        i += 1

    print(f"\n \nNegative Count: {negative_total} | Negative Correct: {negative_correct} \nPositive Total: {positive_total} | Positive Correct: {positive_correct} \nNeutral Total: {neutral_total} | Neutral Correct: {neutral_correct}")
    
    print(f"Negative accuracy: {negative_correct/negative_total}")
    print(f"Neutral accuracy: {neutral_correct/neutral_total}")
    print(f"Positive accuracy: {positive_correct/positive_total}")

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {correct/total}")
    print(f"\nPrecision: {precision}") # TP / TP + FP 
    print(f"Recall: {recall}") # TP / TP + TN
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return correct, total 
        
if __name__ == "__main__":

    print("Loading Model from Hugging Face")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("Importing dataset")
    df = pd.read_csv("/Users/minghill/Desktop/BU/CS505 /CS505_Project/tweet-impact-predictor/data/archive/Tim_dataset.csv")

    print("Conducting inference")
    correct, total = predict(df, tokenizer, model)


    # Accuracy: 0.6557788944723618
