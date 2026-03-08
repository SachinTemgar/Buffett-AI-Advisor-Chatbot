import sys
import os

# Add parent directory to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.financial_data import get_stock_financials
from analysis.buffett_ratios import BuffettAnalyzer

def analyze_stock_for_chatbot(ticker):
    """
    API endpoint for ChatBot to call
    Returns analysis in JSON format
    """
    financials = get_stock_financials(ticker)
    
    if not financials['success']:
        return {'error': f'Could not analyze {ticker}'}
        
    analyzer = BuffettAnalyzer(financials)
    ratios = analyzer.calculate_all_ratios()
    score = analyzer.get_buffett_score(ratios)
    
    return {
        'ticker': ticker,
        'buffett_score': score,
        'ratios': ratios,
        'recommendation': 'BUY' if score > 70 else 'HOLD' if score > 50 else 'AVOID'
    }