import pandas as pd
import numpy as np

class BuffettAnalyzer:
    """
    Calculate all 15 Warren Buffett financial ratios
    Based on the assignment requirements
    """
    
    def __init__(self, financials_dict):
        self.income = financials_dict['income_statement']
        self.balance = financials_dict['balance_sheet']
        self.cashflow = financials_dict['cash_flow']
        self.ticker = financials_dict['ticker']
        
    def calculate_all_ratios(self):
        """Calculate all 15 Warren Buffett ratios"""
        
        results = {
            'income_statement': self._income_statement_ratios(),
            'balance_sheet': self._balance_sheet_ratios(),
            'cash_flow': self._cash_flow_ratios()
        }
        
        return results

    def _income_statement_ratios(self):
        """Calculate 8 income statement ratios"""
        ratios = {}
        
        # 1. Gross Margin
        gross_profit = self._safe_get(self.income, 'Gross Profit')
        total_revenue = self._safe_get(self.income, 'Total Revenue')
        ratios['Gross Margin'] = {
            'value': gross_profit / total_revenue if total_revenue else 0,
            'rule': '> 40%',
            'threshold': 0.40,
            'logic': "Signals the company isn't competing on price"
        }
        
        # 2. SG&A Expense Margin
        sga = self._safe_get(self.income, 'Selling General And Administration')
        ratios['SG&A Expense Margin'] = {
            'value': sga / gross_profit if gross_profit else 0,
            'rule': '< 30%',
            'threshold': 0.30,
            'logic': "Wide-moat companies don't need to spend a lot on overhead"
        }
        
        # 3. R&D Expense Margin
        rd = self._safe_get(self.income, 'Research And Development')
        # If R&D is NaN (not reported), assume 0
        if rd is None: rd = 0
        ratios['R&D Expense Margin'] = {
            'value': rd / gross_profit if gross_profit else 0,
            'rule': '< 30%',
            'threshold': 0.30,
            'logic': "R&D expenses don't always create shareholder value"
        }
        
        # 4. Depreciation Margin
        depreciation = self._safe_get(self.income, 'Reconciled Depreciation')
        # Sometimes listed differently, try generic Depreciation
        if depreciation is None: depreciation = self._safe_get(self.income, 'Depreciation And Amortization')
        
        ratios['Depreciation Margin'] = {
            'value': depreciation / gross_profit if (gross_profit and depreciation) else 0,
            'rule': '< 10%',
            'threshold': 0.10,
            'logic': "Buffett doesn't like businesses that need depreciating assets"
        }
        
        # 5. Interest Expense Margin
        interest = self._safe_get(self.income, 'Interest Expense')
        operating_income = self._safe_get(self.income, 'Operating Income')
        ratios['Interest Expense Margin'] = {
            'value': interest / operating_income if (operating_income and interest) else 0,
            'rule': '< 15%',
            'threshold': 0.15,
            'logic': "Great businesses don't need debt to finance themselves"
        }
        
        # 6. Income Tax Rate
        tax_provision = self._safe_get(self.income, 'Tax Provision')
        pretax_income = self._safe_get(self.income, 'Pretax Income')
        ratios['Income Tax Rate'] = {
            'value': tax_provision / pretax_income if (pretax_income and tax_provision) else 0,
            'rule': 'Current Corporate Tax Rate',
            'threshold': 0.21,
            'logic': "Great businesses are so profitable they pay full tax load"
        }
        
        # 7. Net Margin (Profit Margin)
        net_income = self._safe_get(self.income, 'Net Income')
        ratios['Net Margin'] = {
            'value': net_income / total_revenue if total_revenue else 0,
            'rule': '> 20%',
            'threshold': 0.20,
            'logic': "Great companies convert 20%+ of revenue into net income"
        }
        
        # 8. EPS Growth
        eps_current = self._safe_get(self.income, 'Basic EPS', col=0)
        eps_previous = self._safe_get(self.income, 'Basic EPS', col=1)
        ratios['EPS Growth'] = {
            'value': eps_current / eps_previous if (eps_previous and eps_current) else 0,
            'rule': '> 1 (Positive & Growing)',
            'threshold': 1.0,
            'logic': "Great companies increase profits every year"
        }
        
        return ratios

    def _balance_sheet_ratios(self):
        """Calculate balance sheet ratios"""
        ratios = {}
        
        # 9. Cash > Debt
        cash = self._safe_get(self.balance, 'Cash And Cash Equivalents')
        current_debt = self._safe_get(self.balance, 'Current Debt')
        # If no debt reported, set to small number to avoid div by zero
        if current_debt is None: current_debt = 0.01 
        
        ratios['Cash to Debt Ratio'] = {
            'value': cash / current_debt if cash else 0,
            'rule': '> 1 (More cash than debt)',
            'threshold': 1.0,
            'logic': "Great companies generate cash without needing debt"
        }

        # 10. Return on Equity (ROE) - THE MISSING RATIO (FIXED)
        net_income = self._safe_get(self.income, 'Net Income')
        shareholder_equity = self._safe_get(self.balance, 'Total Stockholder Equity')
        if shareholder_equity is None: 
            shareholder_equity = self._safe_get(self.balance, 'Total Equity Gross Minority Interest')

        ratios['Return on Equity (ROE)'] = {
            'value': net_income / shareholder_equity if (shareholder_equity and net_income) else 0,
            'rule': '> 15%',
            'threshold': 0.15,
            'logic': "Measures how efficiently management uses shareholder capital"
        }
        
        # 11. (Adjusted) Debt to Equity
        total_debt = self._safe_get(self.balance, 'Total Debt')
        total_assets = self._safe_get(self.balance, 'Total Assets')
        
        if total_debt and total_assets:
            shareholder_equity = total_assets - total_debt
            val = total_debt / shareholder_equity if shareholder_equity else 0
        else:
            val = 0
            
        ratios['Adjusted Debt to Equity'] = {
            'value': val,
            'rule': '< 0.80',
            'threshold': 0.80,
            'logic': "Great companies finance themselves with equity"
        }
        
        # 12. Preferred Stock
        # Checking if row exists
        pref_stock = self._safe_get(self.balance, 'Preferred Stock')
        ratios['Preferred Stock'] = {
            'value': 1 if pref_stock else 0, # 1 means exists (bad), 0 means none (good)
            'rule': 'None',
            'threshold': 0, 
            'logic': "Great companies don't fund with preferred stock"
        }
        
        # 13. Retained Earnings Growth
        retained_current = self._safe_get(self.balance, 'Retained Earnings', col=0)
        retained_previous = self._safe_get(self.balance, 'Retained Earnings', col=1)
        
        if retained_current and retained_previous:
            growth = (retained_current - retained_previous) / abs(retained_previous)
        else:
            growth = 0
            
        ratios['Retained Earnings Growth'] = {
            'value': growth,
            'rule': 'Positive Growth',
            'threshold': 0,
            'logic': "Great companies grow retained earnings each year"
        }
        
        # 14. Treasury Stock
        treasury = self._safe_get(self.balance, 'Treasury Shares Number')
        ratios['Treasury Stock'] = {
            'value': 1 if (treasury and treasury != 0) else 0,
            'rule': 'Should Exist',
            'threshold': 0.5, # Threshold logic: >0.5 means it exists (1)
            'logic': "Great companies repurchase their stock"
        }
        
        return ratios

    def _cash_flow_ratios(self):
        """Calculate cash flow ratio"""
        ratios = {}
        
        # 15. CapEx Margin (CapEx / Net Income)
        capex = self._safe_get(self.cashflow, 'Capital Expenditure')
        net_income = self._safe_get(self.cashflow, 'Net Income From Continuing Operations')
        
        if capex and net_income:
            # CapEx is usually negative in cash flow statements, make it positive for ratio
            val = abs(capex) / net_income
        else:
            val = 0
            
        ratios['CapEx Margin'] = {
            'value': val,
            'rule': '< 25%',
            'threshold': 0.25,
            'logic': "Great companies don't need much equipment to generate profits"
        }
        
        return ratios

    def _safe_get(self, df, key, col=0):
        """Safely get value from dataframe"""
        try:
            if key in df.index:
                value = df.loc[key].iloc[col]
                return value if pd.notna(value) else None
            return None
        except:
            return None

    def get_buffett_score(self, ratios):
        """
        Calculate overall Buffett Score (0-100)
        """
        total_criteria = 0
        passed_criteria = 0
        
        for category in ratios.values():
            for ratio_name, ratio_data in category.items():
                if ratio_data['threshold'] is not None and ratio_data['value'] is not None:
                    total_criteria += 1
                    
                    val = ratio_data['value']
                    thresh = ratio_data['threshold']
                    
                    # Logic for "Should NOT Exist" (Preferred Stock)
                    if ratio_name == 'Preferred Stock':
                        if val == 0: passed_criteria += 1
                        
                    # Logic for "Should Exist" (Treasury Stock)
                    elif ratio_name == 'Treasury Stock':
                        if val == 1: passed_criteria += 1
                        
                    # Standard Greater Than Logic
                    elif '>' in ratio_data['rule']:
                        if val > thresh: passed_criteria += 1
                        
                    # Standard Less Than Logic
                    elif '<' in ratio_data['rule']:
                        if val < thresh: passed_criteria += 1
        
        if total_criteria == 0: return 0
        return round((passed_criteria / total_criteria) * 100, 1)