import json
from collections import deque
from datetime import timedelta

import pandas as pd
import streamlit as st


class ProgressBar:
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
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.times = deque()

    def get_update(self, time_delta):
        self.processed_files += 1
        self.times.append(time_delta)

        if len(self.times) < 3:
            return "..."
        if len(self.times) > 15:
            self.times.popleft()

        total_time = sum(self.times, timedelta())
        average_time = total_time / len(self.times)
        remaining_files = self.total_files - self.processed_files
        remaining_time = remaining_files * average_time

        total_seconds = max(int(remaining_time.total_seconds()), 0)
        minutes = total_seconds // 60
        seconds = total_seconds % 60

        return f"{minutes} min {seconds} s"


def download_csv(df: pd.DataFrame):
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    return st.download_button(
        label="CSV",
        data=csv_bytes,
        file_name="sammy_frontend_summary.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
    )


def _clean_result_for_json(item: dict):
    return {
        "filename": item["filename"],
        "class_counts": item["class_counts"],
        "density_info": item["density_info"],
        "stomata": item["stomata"],
    }


def download_metadata(results: list[dict]):
    cleaned = [_clean_result_for_json(item) for item in results]
    json_bytes = json.dumps(cleaned, indent=2).encode("utf-8")

    return st.download_button(
        label="Metadata",
        data=json_bytes,
        file_name="sammy_frontend_metadata.json",
        mime="application/json",
        on_click="ignore",
        icon=":material/download:",
    )
