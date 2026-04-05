# utils/preprocessing.py - NO NLTK DEPENDENCY
"""
Text preprocessing utilities - Pure Python, no external dependencies
"""

import re

def preprocess_text(text):
    """
    EXACT preprocessing function used during training
    MUST match your training preprocessing exactly!
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (keep only letters and spaces)
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def split_sentences(text):
    """
    Split text into sentences using regex
    No NLTK required!
    """
    if not text or not isinstance(text, str):
        return []
    
    # Clean the text first
    text = text.strip()
    if not text:
        return []
    
    # Method 1: Split on sentence endings (.!?) followed by space or end of string
    # This handles most cases
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    
    # If that didn't work well, try simpler split
    if len(sentences) <= 1 and '.' in text:
        # Fallback: split on periods followed by space
        sentences = re.split(r'\.\s+', text)
    
    # Clean up sentences
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        # Remove any trailing punctuation
        s = re.sub(r'[.!?]+$', '', s)
        if s and len(s.split()) >= 2:  # At least 2 words
            cleaned_sentences.append(s)
    
    # If no sentences found, return the whole text as one sentence
    if not cleaned_sentences and text:
        # But only if it's substantial
        if len(text.split()) >= 3:
            cleaned_sentences = [text]
    
    return cleaned_sentences


def clean_response(text):
    """
    Clean user input before processing
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def simple_word_tokenize(text):
    """
    Simple word tokenization (if needed)
    """
    if not text:
        return []
    
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


# Test function (you can remove this in production)
'''def test_tokenizer():
    """Test the sentence splitter"""
    test_texts = [
        "I feel anxious. My heart is racing. I can't breathe.",
        "I'm feeling hopeless and empty. Nothing matters anymore.",
        "I want to kill myself. I have a plan. I bought pills.",
        "I had a good day today. Spent time with family. Feeling hopeful.",
        "This is one long sentence without punctuation so it should stay together"
    ]
    
    print("Testing sentence tokenizer:")
    for text in test_texts:
        sentences = split_sentences(text)
        print(f"\nOriginal: {text}")
        print(f"Split into {len(sentences)} sentences:")
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s}")
    
    return True

# Run test if executed directly
if __name__ == "__main__":
    test_tokenizer()'''