# src/gui/utils.py

from datetime import datetime

def validate_date(date_text):
    try:
        datetime.strptime(date_text, "%d/%m/%Y")
        return True
    except ValueError:
        return False
    
class Stock:
    def __init__(self, ticker, name, start_date, freq):
        self.ticker = ticker
        self.name = name
        self.start_date = start_date
        self.freq = freq

    def __eq__(self, other):
        if not isinstance(other, Stock):
            return NotImplemented
        return self.ticker == other.ticker and self.name == other.name and self.start_date == other.start_date and self.freq == other.freq

    def equals(self, other):
        return self == other

    def __str__(self):
        return f"Stock(ticker={self.ticker}, name={self.name}, start_date={self.start_date}, freq={self.freq})"