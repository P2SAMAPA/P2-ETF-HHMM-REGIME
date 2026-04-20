import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

class USMarketCalendar:
    def __init__(self):
        self.calendar = USFederalHolidayCalendar()
        self.holidays = self.calendar.holidays(start='2000-01-01', end='2030-12-31')
        self.trading_day = CustomBusinessDay(holidays=self.holidays)
    def next_trading_day(self, date=None):
        if date is None: date = pd.Timestamp.today().normalize()
        else: date = pd.Timestamp(date).normalize()
        return date + self.trading_day
