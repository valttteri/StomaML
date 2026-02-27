import json
import tempfile
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import streamlit_ext as ste
from config import API_URL

INFERENCE_COMPLETE = False

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
        self.text = text
        self.width = width
        self.progress_bar = st.progress(value=self.value, text=self.text, width=self.width)
    
    def render(self):
        return self.progress_bar

    def update(self, value: int, text=None):
        self.value = value

        if text is not None:
            self.text = text

        self.progress_bar = self.progress_bar.progress(value=self.value, text=self.text, width=self.width)
        return self.progress_bar


def call_api_with_files(file_paths: list[Path], conf: float = 0.25) -> dict:
    """
    Function for executing inference on an image

    Params:
    file_paths: List with a single image path
    conf: Confidence level for the YOLO model

    Returns:
    Inference results in JSON format
    """
    files_payload = [
        ("files", (p.name, p.read_bytes(), "application/octet-stream"))
        for p in file_paths
    ]
    try:
        resp = requests.post(API_URL, params={"conf": conf}, files=files_payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.exception(e)
        raise


def download_metadata(results):
    """
    Turn the inference metadata into a JSON object, and return a download button
    """
    json_bytes = json.dumps(results, indent=2).encode("utf-8")

    return ste.download_button(
        label="Save metadata",
        data=json_bytes,
        file_name="metadata.json",
        mime="application/json"
    )

def main():
    st.title("Stomatal density app")

    if "inference_done" not in st.session_state:
        st.session_state["inference_done"] = False

    uploaded_files = st.file_uploader(
        "Upload files for analysis",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    # results: The inference metadata
    # file_names: stomata image file names
    # stomata_counts: number of stomata per image
    metadata_arr, filename_arr, density_info_arr = [], [], []

    if st.button("Run analysis") and uploaded_files:
        st.session_state["inference_done"] = False
        # Initialize a progress bar
        progress_bar = ProgressBar(0, "Inference in progress...", "stretch")
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
                    api_json = call_api_with_files(temp_path, conf=0.25)

                    response_data = api_json["results"][0]
                    filename = uf.name
                    density_info = list(response_data["density_info"].values())
                    metadata = response_data["stomata"]

                    metadata_arr.append(metadata)
                    filename_arr.append(filename)
                    density_info_arr.append(density_info)


                    # Update the progress bar
                    progress_bar.update(value=ratio)

            progress_bar.update(value=100, text="Inference Finished!")
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

        # CSV ready for download
        csv_bytes = df.to_csv(index=True).encode("utf-8")
        ste.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="stoma_results.csv",
            mime="text/csv",
        )

        # Button for downloading metadata
        download_metadata(results=metadata_arr)

if __name__ == "__main__":
    main()
