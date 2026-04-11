import pandas as pd
import joblib
from nltk.tokenize import sent_tokenize
import nltk

# Download tokenizer (run once)


# nltk.download('punkt')
# nltk.download('punkt_tab')

# =========================
# 1. LOAD MODEL + VECTORIZER
# =========================
model = joblib.load("psychotherapy_svm_model.pkl")


# =========================
# 2. LOAD YOUR RAW DATASET
# =========================
# Change column name if needed
df = pd.read_excel("mental_health_combined_test.xlsx")

# Assume column name is 'text'
TEXT_COLUMN = "text"

# =========================
# 3. FUNCTION: SPLIT + PREDICT
# =========================
def process_text(text):
    sentences = sent_tokenize(str(text))
    
    results = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 3:
            continue  # skip very short/noisy sentences
        
        #vec = vectorizer.transform([sentence])
        
        pred = model.predict([sentence])[0]
        prob = model.predict_proba([sentence]).max()
        
        results.append({
            "sentence": sentence,
            "predicted_label": pred,
            "confidence": round(float(prob), 3),
            "final_label": "",   # YOU will fill this manually
            "review": "YES" if prob < 0.6 else "NO"
        })
    
    return results

# =========================
# 4. APPLY TO DATASET
# =========================
all_rows = []

for idx, row in df.iterrows():
    text = row[TEXT_COLUMN]
    
    processed = process_text(text)
    
    for item in processed:
        all_rows.append(item)

# =========================
# 5. CREATE NEW DATAFRAME
# =========================
new_df = pd.DataFrame(all_rows)

# =========================
# 6. SAVE FOR MANUAL REVIEW
# =========================
new_df.to_csv("sentence_level_labeled.csv", index=False)

print("✅ Done! File saved as 'sentence_level_labeled.csv'")
