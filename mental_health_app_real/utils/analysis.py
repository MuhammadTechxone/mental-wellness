"""
Core analysis engine: sentence-level prediction and aggregation
"""

import pandas as pd
import numpy as np
from collections import Counter
from .preprocessing import preprocess_text, split_sentences


class MentalHealthAnalyzer:
    """
    Handles all analysis logic for mental health predictions
    """
    
    def __init__(self, model, classes):
        """
        Initialize with loaded pipeline model
        """
        self.model = model
        self.classes = classes
        self.sentence_predictions = []
        self.aggregated_results = {}
        
    def analyze_response(self, text, question_id=None):
        """
        Analyze a single response by breaking into sentences
        Returns list of sentence-level predictions
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = split_sentences(text)
        results = []
        
        for sentence in sentences:
            if len(sentence.split()) < 2:  # Skip very short sentences
                continue
                
            # Preprocess (CRITICAL: must match training)
            cleaned = preprocess_text(sentence)
            
            # Predict using pipeline (handles TF-IDF automatically)
            prediction = self.model.predict([cleaned])[0]
            
            # Get prediction confidence (probabilities)
            try:
                decision_scores = self.model.decision_function([cleaned])[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probs = exp_scores / exp_scores.sum()
                confidence = probs[list(self.model.classes_).index(prediction)]
            except:
                confidence = None
            
            results.append({
                'sentence': sentence,
                'prediction': prediction,
                'confidence': confidence,
                'question_id': question_id,
                'cleaned': cleaned
            })
        
        return results
    
    def analyze_all_responses(self, responses_dict):
        """
        Analyze all questionnaire responses
        responses_dict: {question_id: response_text}
        """
        all_predictions = []
        
        for q_id, response in responses_dict.items():
            if response and response.strip():
                sentence_results = self.analyze_response(response, q_id)
                all_predictions.extend(sentence_results)
        
        self.sentence_predictions = all_predictions
        self.aggregated_results = self._aggregate_predictions()
        
        return self.sentence_predictions
    
    def _aggregate_predictions(self):
        """
        Aggregate all sentence predictions into summary statistics
        """
        if not self.sentence_predictions:
            return {}
        
        # Count predictions
        pred_counts = Counter([p['prediction'] for p in self.sentence_predictions])
        total = len(self.sentence_predictions)
        
        # Calculate percentages
        distribution = {}
        for class_name in self.classes:
            count = pred_counts.get(class_name, 0)
            distribution[class_name] = (count / total * 100) if total > 0 else 0
        
        # Find primary and secondary indicators
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'total_sentences': total,
            'distribution': distribution,
            'primary_indicator': sorted_classes[0][0] if sorted_classes else None,
            'primary_percentage': sorted_classes[0][1] if sorted_classes else 0,
            'secondary_indicator': sorted_classes[1][0] if len(sorted_classes) > 1 else None,
            'secondary_percentage': sorted_classes[1][1] if len(sorted_classes) > 1 else 0,
            'raw_predictions': self.sentence_predictions
        }
        
        return result
    
    def get_risk_level(self):
        """
        Determine risk level based on aggregated predictions
        """
        if not self.aggregated_results:
            return "unknown"
        
        dist = self.aggregated_results['distribution']
        suicidal_pct = dist.get('suicidal', 0)
        
        # Import thresholds from config
        from config.settings import RISK_THRESHOLDS as TH
        
        if suicidal_pct >= TH['suicidal_high_risk']:
            return "high"
        elif suicidal_pct >= TH['suicidal_moderate']:
            return "moderate_high"
        elif (dist.get('depression', 0) + dist.get('anxiety', 0)) >= TH['depression_anxiety_threshold']:
            return "moderate"
        elif dist.get('normal', 0) >= 60:
            return "low"
        else:
            return "mild"
    
    def get_suicidal_flagged_sentences(self):
        """
        Return all sentences flagged as suicidal for review
        """
        if not self.sentence_predictions:
            return []
        
        return [p for p in self.sentence_predictions if p['prediction'] == 'suicidal']


def create_distribution_dataframe(distribution):
    """
    Create a formatted DataFrame for visualization
    """
    df = pd.DataFrame({
        'Mental Health Indicator': list(distribution.keys()),
        'Percentage': list(distribution.values())
    })
    return df.sort_values('Percentage', ascending=False)