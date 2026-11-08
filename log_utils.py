import numpy as np
from datetime import datetime

class run_time:
    """ Usage:
    run_time_calc = run_time()
    run_time_calc.print_time()
    """
    def __init__(self):
        self.start_time = datetime.now()
    
    def block_end(self):
        self.end_time = datetime.now()
        self.diff = self.end_time - self.start_time
        self.days = self.diff.days
        self.hours = self.diff.seconds // 3600
        self.minutes = (self.diff.seconds % 3600) // 60
        self.seconds = self.diff.seconds % 60
        self.microseconds = self.diff.microseconds
    
    def print_time(self, n=20):
        dashes = "-" * (n)
        print(f"\n{dashes}\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTotal execution time:")
        day_txt = "day" if self.days == 1 else "days"
        hour_txt = "hour" if self.hours == 1 else "hours"
        minute_txt = "minute" if self.minutes == 1 else "minutes"
        second_txt = "second" if self.seconds == 1 else "seconds"
        microsecond_txt = "microsecond" if self.microseconds == 1 else "microseconds"
        print(f"{self.days} {day_txt}, {self.hours} {hour_txt}, {self.minutes} {minute_txt}, {self.seconds} {second_txt}, and {self.microseconds} {microsecond_txt}\n{dashes}")

def print_info(n=20, **kwargs):
    dashes = "-" * n
    
    print(f"\n{dashes}\nMax Score: {max_score:.1f} | Episode: {max_ep}\n{dashes}\n")

def calculate_stats(metric_data, metric="metric"):
    mean_data = np.mean(metric_data)
    std_data = np.std(metric_data)
    return "{}: {} ± {}".format(metric, np.round(mean_data, 4), np.round(std_data, 4))        