import yfinance as yf
import pandas as pd

def get_stock_financials(ticker):
    """
    Fetch all three financial statements for a given ticker.
    Returns: dict with income_statement, balance_sheet, cash_flow
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Force fetching data to ensure it exists
        # Note: yfinance loads lazily, accessing properties triggers the download
        income = stock.financials
        balance = stock.balancesheet
        cashflow = stock.cashflow
        
        if income.empty:
            return {'success': False, 'error': "No financial data found"}

        return {
            'ticker': ticker,
            'info': stock.info,
            'income_statement': income,
            'balance_sheet': balance,
            'cash_flow': cashflow,
            'success': True
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }

def get_company_info(ticker):
    """Get basic company information including LIVE PRICE"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try different keys because Yahoo Finance API varies sometimes
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No summary available.'),
            'current_price': current_price # <--- NEW FIELD ADDED
        }
    except:
        return {
            'name': ticker, 
            'sector': 'N/A', 
            'industry': 'N/A', 
            'market_cap': 'N/A',
            'summary': 'N/A',
            'current_price': 'N/A'
        }