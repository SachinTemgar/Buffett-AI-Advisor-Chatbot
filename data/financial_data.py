import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

def create_robust_session():
    """Create a session with retry logic and proper headers"""
    session = requests.Session()
    
    # Add comprehensive headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })
    
    # Add retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_stock_financials(ticker):
    """
    Fetch all three financial statements for a given ticker.
    Returns: dict with income_statement, balance_sheet, cash_flow
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Create robust session
            session = create_robust_session()
            
            # Create ticker with session
            stock = yf.Ticker(ticker, session=session)
            
            # Add small delay to avoid rate limiting
            if attempt > 0:
                time.sleep(3)
            
            # Fetch data
            income = stock.financials
            balance = stock.balancesheet
            cashflow = stock.cashflow
            
            # Verify data
            if income is None or income.empty:
                if attempt < max_attempts - 1:
                    time.sleep(3)
                    continue
                return {
                    'success': False,
                    'error': f"No financial data available for {ticker}"
                }
            
            return {
                'ticker': ticker,
                'info': stock.info,
                'income_statement': income,
                'balance_sheet': balance,
                'cash_flow': cashflow,
                'success': True
            }
            
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(3)
                continue
            return {
                'ticker': ticker,
                'success': False,
                'error': f"Error: {str(e)}"
            }
    
    return {
        'ticker': ticker,
        'success': False,
        'error': "Failed after multiple attempts. Please try again later."
    }

def get_company_info(ticker):
    """Get basic company information including LIVE PRICE"""
    try:
        session = create_robust_session()
        stock = yf.Ticker(ticker, session=session)
        info = stock.info
        
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
