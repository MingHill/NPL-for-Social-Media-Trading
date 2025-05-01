from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

def analyze_sentiment(text, tokenizer, model):
    '''
        Processes a single tweet and predicts if it is Negative(0), Neutral(1) or Positive(2)
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
    
    for index, row in df.iterrows():
        text = row['text']
        actual_label = row['label']
        
        predicted_sentiment, confidence = analyze_sentiment(text, tokenizer, model)
        sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        predicted_label = sentiment_mapping[predicted_sentiment]
        if predicted_label == actual_label: 
            correct += 1 
        print(f"{i}: Predicted Label: {predicted_label} | Actual Label:{actual_label}")
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
    
    print(f" Negative Count: {negative_total} | Negative Correct: {negative_correct} \n Positive Total: {positive_total} | Positive Correct: {positive_correct} \n Neutral Total: {neutral_total} | Neutral Correct: {neutral_correct}")
    
    # print(f"Negative accuracy: {negative_correct/negative_total}")
    # print(f"Neutral accuracy: {neutral_correct/neutral_total}")
    # print(f"Positive accuracy: {positive_correct/positive_total}")
    return correct, total,  

if __name__ == "__main__":

    print("Loading Model from Hugging Face")
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Define 3 labels for sentiment

    print("Importing dataset")
    df = pd.read_csv("/Users/minghill/Desktop/BU/CS505 /CS505_Project/tweet-impact-predictor/data/archive/cleaned_send_train.csv")  # Replace with your file path

    print("Conducting inference")
    correct, total = predict(df, tokenizer, model)

    print(f"Accuracy: {correct/total}") 
    # Accuracy: 0.7039363484087102
