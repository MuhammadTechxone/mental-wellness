# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# For reproducible results
np.random.seed(42)

# ------------------------------
# 1. Load and Prepare the Data
# ------------------------------
# Assuming you have a CSV with 'text' and 'status' columns
# Example structure:
# df = pd.read_csv('therapy_data.csv')
# print(df.head())
# >>> "I feel so hopeless today" , "depression"
# >>> "My heart is racing constantly", "anxiety"

# Create sample data (replace this with your actual data loading)
'''def create_sample_data():
    """Creates a small sample dataset for demonstration"""
    data = {
        'text': [
            # Depression examples
            "I feel so empty inside",
            "Nothing brings me joy anymore",
            "I can't get out of bed",
            "Life feels meaningless",
            "I'm always tired and sad",
            # Anxiety examples
            "My heart is pounding",
            "I can't stop worrying",
            "I feel like something bad will happen",
            "My mind is racing",
            "I feel tense all the time",
            # Suicidal examples
            "I want to end it all",
            "Life isn't worth living",
            "I wish I could just disappear",
            "Everyone would be better off without me",
            "I'm planning to kill myself",
            # Normal examples
            "I had a good day today",
            "I enjoyed talking with my friend",
            "Work was productive",
            "I'm feeling okay",
            "Looking forward to the weekend"
        ],
        'status': [
            'depression', 'depression', 'depression', 'depression', 'depression',
            'anxiety', 'anxiety', 'anxiety', 'anxiety', 'anxiety',
            'suicidal', 'suicidal', 'suicidal', 'suicidal', 'suicidal',
            'normal', 'normal', 'normal', 'normal', 'normal'
        ]
    }
    return pd.DataFrame(data)
# synthetic_therapy_data.csv
# synthetic_mental_health_data_enhanced.csv
# Load your actual data here
# df = pd.read_csv('your_therapy_data.csv')
df = create_sample_data()'''# Using sample data for demonstration
df = pd.read_csv("mental_health_BALANCED.csv")
df = df.dropna(subset=["text"])
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['status'].value_counts())
print(f"\nFirst few rows:")
print(df.head())

# ------------------------------------
# 2. Text Preprocessing and Vectorization
# ------------------------------------
# Create a preprocessing function
def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits (keep only letters and spaces)
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# ------------------------------------
# 3. Split Data into Train and Test Sets
# ------------------------------------
X = df['clean_text']
y = df['status']

# With very small dataset, use a larger test split or just do cross-validation
# Option 1: Use a smaller test set (e.g., 20% of data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing (4 samples)
    random_state=42,
    stratify=y  # Important: maintains class balance
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ------------------------------------
# 4. Build the Pipeline with Adjusted Parameters for Small Data
# ------------------------------------
# Create a pipeline with simplified parameters for small dataset
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,  # Reduced from 5000 for small data
        ngram_range=(1, 2),  # Just unigrams for small data (bigrams might be too sparse)
        stop_words='english',
        min_df=1,  # Allow words that appear in just 1 document
        max_df=1.0  # Don't filter out common words
    )),
    ('clf', LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=2000,
        dual='auto'  # Handle small datasets better
    ))
])

# ------------------------------------
# 5. Simplified Hyperparameter Tuning for Small Data
# ------------------------------------
# Instead of GridSearchCV with folds, let's do a simple manual approach
# or use cross-validation with fewer folds

# Option A: Manual parameter selection (recommended for very small data)
print("\nTraining with default parameters...")
pipeline.fit(X_train, y_train)
train_accuracy = pipeline.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy:.3f}")

# Option B: Cross-validation with 3 folds instead of 5
print("\nPerforming 3-fold cross-validation...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Use the pipeline as is (with default parameters for now)
best_model = pipeline

# ------------------------------------
# 6. Evaluate on Test Set
# ------------------------------------
# Make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.3f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ------------------------------------
# 7. Feature Importance (What words drive decisions?)
# ------------------------------------
# Extract the SVM coefficients
feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
coefficients = best_model.named_steps['clf'].coef_

# Show top words for each class
print("\nTop 10 words driving each classification:")
for i, class_label in enumerate(best_model.classes_):
    # Get coefficients for this class
    coef = coefficients[i]
    # Get indices of top 10 positive coefficients
    top_indices = np.argsort(coef)[-10:][::-1]
    top_words = [(feature_names[j], coef[j]) for j in top_indices if j < len(feature_names)]
    
    print(f"\n{class_label.upper()}:")
    for word, score in top_words[:5]:  # Show top 5 for small dataset
        print(f"  {word}: {score:.3f}")

# ------------------------------------
# 8. Function for Real-time Prediction
# ------------------------------------
def predict_patient_statement(statement, model=best_model):
    """
    Predict the status of a single patient statement.
    This is what you'd use during a therapy session.
    """
    # Preprocess
    clean_statement = preprocess_text(statement)
    
    # Predict
    prediction = model.predict([clean_statement])[0]
    
    # Get decision scores
    decision_scores = model.decision_function([clean_statement])[0]
    
    # Convert to pseudo-probabilities (softmax)
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / exp_scores.sum()
    
    # Create result dictionary
    results = {
        'statement': statement,
        'prediction': prediction,
        'probabilities': dict(zip(model.classes_, probs))
    }
    
    return results

# Test the prediction function
test_statements = [
    "I can't stop thinking about dying",
    "My heart is racing and I feel scared",
    "I feel completely empty inside",
    "I had a nice lunch with my family",
    "I want to kill myself tonight",  # Additional test
    "My anxiety is through the roof",
    "I feel like giving up",
"I can't take this anymore",
"What's the point",
"I'm so tired of fighting",
"I feel hopeless but I won't hurt myself",
"I'm anxious about dying",
"Death feels like the only peace",
"I think about death sometimes but I'm not suicidal",
"I want the pain to stop but I don't want to die",
"I wish I could disappear for a while",
"Everyone would be better off without me here",

    
]

print("\n" + "="*50)
print("TESTING REAL-TIME PREDICTIONS")
print("="*50)

for statement in test_statements:
    result = predict_patient_statement(statement)
    print(f"\nStatement: '{result['statement']}'")
    print(f"Prediction: {result['prediction']}")
    print("Probabilities:")
    # Sort by probability for better readability
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for status, prob in sorted_probs:
        print(f"  {status}: {prob:.2%}")
    print("-"*30)

# ------------------------------------
# 9. Save the Model for Later Use
# ------------------------------------
import joblib

# Save the model
model_filename = 'psychotherapy_svm_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\nModel saved as '{model_filename}'")

# To load it later:
# loaded_model = joblib.load('psychotherapy_svm_model.pkl')

# ------------------------------------
# 10. Practical Advice for Small Datasets
# ------------------------------------
print("\n" + "="*50)
print("PRACTICAL ADVICE FOR YOUR 1000-ROW DATASET")
print("="*50)
print("""
When you get your full 1000-row dataset:

1. **Increase test size**: Use test_size=0.2 (200 samples for testing)
2. **Use GridSearchCV**: With your full dataset, you can use the original GridSearchCV
3. **Enable bigrams**: ngram_range=(1, 2) will work better with more data
4. **More features**: Increase max_features to 5000-10000
5. **Focus on recall for 'suicidal'**: In mental health, missing a suicidal case is worse than false alarm

The current code is adjusted for the small sample, but will work perfectly with your real data!
""")