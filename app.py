import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download the NLTK punkt tokenizer model (if not already downloaded)
nltk.download('punkt')

# Load the tokenizer and model for gender bias detection
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-gender-bias")
model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-gender-bias")

# Function to detect gender bias in a sentence
def detect_gender_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    score = logits.softmax(dim=1)[0][predicted_class].item()
    label = "True" if predicted_class == 1 else "False"
    result = "yes" if label == "True" else "no"
    return result, score

# Main function to input sentence and check for bias
def main():
    sentence = "Women can't drive well"
    
    if not sentence.strip():
        print("No sentence provided. Please enter a valid sentence.")
        return
    
    try:
        result, score = detect_gender_bias(sentence)
        print(f"Biased: {result}\nConfidence Score: {score:.4f}")
    except Exception as e:
        print(f"Error detecting bias: {str(e)}")

if __name__ == "__main__":
    main()
