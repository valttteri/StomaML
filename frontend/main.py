import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import requests
import streamlit as st
import streamlit_ext as ste
from config import API_URL

class ProgressBar:
    """
    Class for inference progress bar

    Params:
    value: 0 <= x <= 100
    text: a string displayed on top of the progress bar
    width: "stretch" or int
    """
    def __init__(self, value: int, text: str, width: str):
        self.value = value
        self.text = ""
        self.width = width
        self.progress_bar = st.progress(value=self.value, text=self.text, width=self.width)
    
    def render(self):
        return self.progress_bar

    def update(self, value: int, text=None):
        self.value = value

        if text is not None:
            #self.text = text
            self.text = ""

        self.progress_bar = self.progress_bar.progress(value=self.value, text=self.text, width=self.width)
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


def call_api_with_files(
        file_paths: list[Path],
        conf: float = 0.25,
        scale: float = 10/46
    ) -> dict:
    """
    Function for executing inference on an image

    Params:
    file_paths: List with a single image path
    conf: Confidence level for the YOLO model
    scale: Image scale in micrometers

    Returns:
    Inference results in JSON format
    """
    files_payload = [
        ("files", (p.name, p.read_bytes(), "application/octet-stream"))
        for p in file_paths
    ]
    try:
        resp = requests.post(API_URL, params={"conf": conf, "scale": scale}, files=files_payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.exception(e)
        raise


def download_csv(df: pd.DataFrame):
    """
    Button for downloading a csv file with stomatal density info etc.

    Params:
    df: a dataframe containing all relevant data
    """
    csv_bytes = df.to_csv(index=True).encode("utf-8")

    return ste.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="stoma_results.csv",
        mime="text/csv",
    )

def download_metadata(results: list):
    """
    Turn the inference metadata into a JSON object, and return a download button

    Params:
    results: a list containing metadata of each individual stomata
    """
    json_bytes = json.dumps(results, indent=2).encode("utf-8")

    return ste.download_button(
        label="Download Metadata",
        data=json_bytes,
        file_name="metadata.json",
        mime="application/json"
    )

def main():
    st.title("StomaML")

    if "inference_done" not in st.session_state:
        st.session_state["inference_done"] = False

    uploaded_files = st.file_uploader(
        "Upload files for analysis",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    # metadata_arr: The inference metadata
    # filename_arr: stomata image file names
    # density_info_arr: data to be included in the csv file
    metadata_arr, filename_arr, density_info_arr = [], [], []

    # Slider for defining image scale in um / pixel
    scale_slider = st.slider(
        label="Image Scale",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        format="%d $\mu$m",
        width=200
    )
    scale_value = scale_slider / 46

    if st.button("Run analysis") and uploaded_files:
        st.session_state["inference_done"] = False
        # Initialize a progress bar
        time_approximator = RemainingTime(total_files=len(uploaded_files))
        inference_status = st.empty()
        progress_bar = ProgressBar(0, "Inference in progress. Time remaining:", "stretch")
        progress_bar.render()


        # Create a temporary directory for Stomata images
        with tempfile.TemporaryDirectory() as tmpdir:

            for idx, uf in enumerate(uploaded_files):
                # Ratio is used for rendering the progress bar
                ratio = int(round(idx/len(uploaded_files), 2)*100)

                suffix = Path(uf.name).suffix or ".png"
                with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=suffix, delete=False) as tf:
                    # Create a temp path for the current file (image)
                    temp_path = []
                    tf.write(uf.getvalue())
                    temp_path.append(Path(tf.name))

                    # api_json contains inference results of a single image
                    start = datetime.now()
                    api_json = call_api_with_files(temp_path, conf=0.25, scale=scale_value)
                    end = datetime.now()
                    
                    total_time = end - start
                    time_left = time_approximator.get_update(time=total_time) # Get an approximation of remaining inference time

                    response_data = api_json["results"][0]
                    filename = uf.name
                    density_info = list(response_data["density_info"].values())
                    metadata = response_data["stomata"]

                    # Save data from a single inference result
                    metadata_arr.append(metadata)
                    filename_arr.append(filename)
                    density_info_arr.append(density_info)

                    # Update the progress bar and inference status
                    progress_bar.update(value=ratio)
                    inference_status.markdown(f"Inference in progress<br>Time remaining: {time_left}", unsafe_allow_html=True)

            progress_bar.update(value=100)
            inference_status.write("Inference done!")
            st.session_state["inference_done"] = True

    # Display inference results once they are ready
    if st.session_state["inference_done"]:
        #st.json(results)
        st.subheader("Results table")

        df_cols = [
            "total_pixels",
            "non_countable_pixels",
            "countable_pixels",
            "stomata_count",
            "stomatal_density_per_px2",
            "um_per_px",
            "stomatal_density_per_mm2"
        ]

        df = pd.DataFrame(data=density_info_arr, columns=df_cols)
        df["Filename"] = filename_arr
        df = df.set_index("Filename")

        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(spec=[1, 1], width=250)

        # Button for downloading csv
        with col1:
            download_csv(df=df)

        # Button for downloading metadata
        with col2:
            download_metadata(results=metadata_arr)

if __name__ == "__main__":
    main()
