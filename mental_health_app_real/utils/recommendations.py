"""
Recommendation engine based on risk assessment
"""

from config.settings import RECOMMENDATIONS


class RecommendationEngine:
    """
    Provides appropriate recommendations based on analysis results
    """
    
    def __init__(self):
        self.recommendations = RECOMMENDATIONS
    
    def get_recommendations(self, primary_indicator, risk_level, suicidal_flagged=None):
        """
        Get tailored recommendations based on primary indicator and risk level
        """
        # Handle high-risk suicidal cases first
        if risk_level in ['high', 'moderate_high'] and suicidal_flagged:
            return self.recommendations['suicidal']
        
        # Get recommendations based on primary indicator
        if primary_indicator in self.recommendations:
            return self.recommendations[primary_indicator]
        
        # Default to normal if unknown
        return self.recommendations['normal']
    
    def get_crisis_resources(self):
        """
        Return crisis resources for high-risk situations
        """
        return {
            'title': "🚨 Crisis Resources",
            'resources': [
                "**National Suicide Prevention Lifeline:** 988",
                "**Crisis Text Line:** Text HOME to 741741",
                "**Veterans Crisis Line:** 988 then press 1",
                "**Trevor Project (LGBTQ+):** 1-866-488-7386",
                "**SAMHSA National Helpline:** 1-800-662-4357"
            ],
            'message': "These resources provide immediate, confidential support 24/7."
        }
    
    def format_recommendations(self, recommendations_dict):
        """
        Format recommendations for display
        """
        if not recommendations_dict:
            return ""
        
        formatted = f"### {recommendations_dict['title']}\n\n"
        formatted += f"{recommendations_dict['message']}\n\n"
        
        for i, tip in enumerate(recommendations_dict['tips'], 1):
            formatted += f"{i}. {tip}\n"
        
        return formatted