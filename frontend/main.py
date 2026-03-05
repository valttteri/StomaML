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
from utils import RemainingTime, ProgressBar

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

    return st.download_button(
        label="CSV",
        data=csv_bytes,
        file_name="stoma_results.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:"
    )

def download_metadata(results: list):
    """
    Turn the inference metadata into a JSON object, and return a download button

    Params:
    results: a list containing metadata of each individual stomata
    """
    json_bytes = json.dumps(results, indent=2).encode("utf-8")

    return st.download_button(
        label="Metadata",
        data=json_bytes,
        file_name="metadata.json",
        mime="application/json",
        on_click="ignore",
        icon=":material/download:"
    )

def main():
    st.title("StomaML")

    if "inference_done" not in st.session_state:
        st.session_state["inference_done"] = False

    if "file_uploader" not in st.session_state: # State variable used for resetting the file uploader
        st.session_state["file_uploader"] = 1

    uploaded_files = st.file_uploader(
        "Upload files for analysis",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key=st.session_state["file_uploader"]
    )

    # metadata_arr: The inference metadata
    # filename_arr: stomata image file names
    # density_info_arr: data to be included in the csv file
    metadata_arr, filename_arr, density_info_arr = [], [], []

    # Columns for Image scale, Files uploaded and Clear files
    col1, col2 = st.columns(spec=[1, 1], gap="medium", width=600) 

    # Slider for defining image scale in um / pixel
    with col1:
        scale_value = st.number_input(
            label="Image scale ($\mu$*m* / pixel)",
            min_value=0.0000000001,
            max_value=10.0,
            value=10/46,
            step=0.0000000001,
            format="%0.10f",
            placeholder="Enter value"
        )

    with col2:
        # Subcolumns for better positioning of components
        sub_col1, sub_col2 = st.columns([1, 2])
        with sub_col1:
            st.metric(label="Files uploaded", value=len(uploaded_files)) # Display the number of uploaded files
        with sub_col2:
            if len(uploaded_files) > 0 and st.button(label="Reset", icon=":material/delete:"):
                st.session_state["file_uploader"] += 1
                st.rerun()
    

    if st.button("Run analysis", icon=":material/bolt:") and uploaded_files:
        st.session_state["inference_done"] = False

        # Initialize an inference time approximator
        time_approximator = RemainingTime(total_files=len(uploaded_files))

        # Message displayed on top of the progress bar
        inference_status = st.empty()
        inference_status.markdown("Analysis in progress<br>Time remaining: ...", unsafe_allow_html=True)

        # Initialize a progress bar
        progress_bar = ProgressBar(0, "stretch")
        progress_bar.render()


        # Create a temporary directory for Stomata images
        with tempfile.TemporaryDirectory() as tmpdir:

            for idx, uf in enumerate(uploaded_files):
                # Ratio is used for rendering the progress bar
                ratio = int(round((idx+1)/len(uploaded_files), 2)*100)

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

                    inference_time = end - start
                    time_left = time_approximator.get_update(time=inference_time) # Get an approximation of remaining inference time

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
                    inference_status.markdown(f"Analysis in progress<br>Time remaining: {time_left}", unsafe_allow_html=True)

            progress_bar.update(value=100)
            inference_status.markdown("Analysis done!<br>Time remaining: 0 min 0 s", unsafe_allow_html=True)
            st.session_state["inference_done"] = True
    

    # Display inference results once they are ready
    if st.session_state["inference_done"]:
        #st.json(results)
        st.subheader("Analysis results")

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

        col1, col2 = st.columns(spec=[0.35, 0.65], gap="small", width=250) # Columns for download buttons

        # Button for downloading csv
        with col1:
            download_csv(df=df)

        # Button for downloading metadata
        with col2:
            download_metadata(results=metadata_arr)

if __name__ == "__main__":
    main()
