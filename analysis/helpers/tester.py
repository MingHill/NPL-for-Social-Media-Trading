import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def preprocess_tweet(text):
    """Preprocess tweet by replacing usernames and URLs with placeholders"""
    words = []
    for word in text.split():
        if word.startswith('@'):
            words.append('@user')
        elif word.startswith('http'):
            words.append('http')
        else:
            words.append(word)
    return ' '.join(words)

def process_text(text, tokenizer):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    # Convert dict of tensors to tensors and remove batch dimension
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

def get_sentiment(text, tokenizer, model):
    inputs = process_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu()   # [num_labels]
        pred   = torch.argmax(probs).item()           # int

    # Optional: map id → label if the model has that info
    MAP = {0: "neutral", 1: "positive", 2: "negative"}
    return probs.numpy()[pred], MAP[pred], probs.numpy()

tokenizer = AutoTokenizer.from_pretrained("models/sentiment")
model = AutoModelForSequenceClassification.from_pretrained("models/sentiment")

sample_texts = [
    """Huge Print $NVDA Size: 1075008 Price: 222.42 Time: 1600 Amount: $239,103,279.36 - Delayed - For real time prints subscribe to  
Runners 📈: 
Losers 📉: 
Gappers 🪜:""",
    """Spotify's More Confident Than Ever in Its Podcast Strategy""",
    """I'm not sure what to think about this product.""",
]

for text in sample_texts:
    label_probs, label, probs_array = get_sentiment(preprocess_tweet(text), tokenizer, model)
    print(f"Text: {text}")
    print(f"Label: {label}")
    print(f"Probs: {label_probs}")
    print(f"Probs Array: {probs_array}")
    print("-"*50)