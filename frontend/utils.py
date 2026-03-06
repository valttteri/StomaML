from datetime import datetime, timedelta
from collections import deque
import streamlit as st

class ProgressBar:
    """
    Class for inference progress bar

    Params:
    value: 0 <= x <= 100
    text: a string displayed on top of the progress bar
    width: "stretch" or int
    """
    def __init__(self, value: int, width: str):
        self.value = value
        self.width = width
        self.progress_bar = st.progress(value=self.value, width=self.width)
    
    def render(self):
        return self.progress_bar

    def update(self, value: int):
        self.value = value

        self.progress_bar = self.progress_bar.progress(value=self.value, width=self.width)
        return self.progress_bar
    
class RemainingTime:
    """
    Class for approximating the remaining inference time

    Params:
    total_files: the number of uploaded files
    """
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.times = deque()

    def get_update(self, time: datetime):
        """
        Update the remaining time approximation

        Params:
        time: The inference time of a single image
        """
        self.processed_files += 1
        self.times.append(time)
        
        if len(self.times) < 5: # Don't return an approximation immediately
            return "..."
        if len(self.times) > 15: # Only keep track of 15 latest inference times
            self.times.popleft()

        total_time = sum(self.times, timedelta())
        
        average_time = total_time / len(self.times)
        remaining_files = self.total_files - self.processed_files
        
        # An approximation of how much longer the inference will take
        remaining_time = remaining_files * average_time

        # Format remaining time as X min Y s
        total_seconds = remaining_time.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)

        formatted_time = f"{minutes} min {seconds} s"

        return formatted_time
