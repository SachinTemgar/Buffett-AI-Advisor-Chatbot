import yfinance as yf
import pandas as pd
import requests
import time

def get_stock_financials(ticker):
    """
    Fetch all three financial statements for a given ticker.
    Returns: dict with income_statement, balance_sheet, cash_flow
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Create session with headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            })
            
            # Use download method for better reliability
            stock = yf.Ticker(ticker, session=session)
            
            # Try to get data
            income = stock.financials
            balance = stock.balancesheet
            cashflow = stock.cashflow
            info = stock.info
            
            # Verify data exists
            if income is None or income.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {
                    'success': False, 
                    'error': f"No financial data found for {ticker}. The ticker may be invalid or data temporarily unavailable."
                }
            
            return {
                'ticker': ticker,
                'info': info,
                'income_statement': income,
                'balance_sheet': balance,
                'cash_flow': cashflow,
                'success': True
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                'ticker': ticker,
                'success': False,
                'error': f"Error retrieving data: {str(e)}"
            }
    
    return {
        'ticker': ticker,
        'success': False,
        'error': "Failed after multiple retry attempts"
    }

def get_company_info(ticker):
    """Get basic company information including LIVE PRICE"""
    try:
        # Create session with headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker, session=session)
        info = stock.info
        
        # Try different keys because Yahoo Finance API varies sometimes
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No summary available.'),
            'current_price': current_price
        }
    except Exception as e:
        return {
            'name': ticker, 
            'sector': 'N/A', 
            'industry': 'N/A', 
            'market_cap': 'N/A',
            'summary': 'N/A',
            'current_price': 'N/A'
        }
