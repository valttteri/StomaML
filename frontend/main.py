import json
import tempfile
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from config import API_URL


def call_api_with_files(file_paths: list[Path], conf: float = 0.25) -> dict:
    files_payload = [
        ("files", (p.name, p.read_bytes(), "application/octet-stream"))
        for p in file_paths
    ]
    try:
        resp = requests.post(API_URL, params={"conf": conf}, files=files_payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise Exception



def main():
    st.title("Stomatal density app")
    uploaded_files = st.file_uploader(
        "Upload files for analysis",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Run analysis") and uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_paths = []

            for uf in uploaded_files:
                suffix = Path(uf.name).suffix or ".png"
                with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=suffix, delete=False) as tf:
                    tf.write(uf.getvalue())
                    temp_paths.append(Path(tf.name))

            #api_json = call_api_with_files(temp_paths, conf=0.25)

        #st.json(api_json)

        st.success("Done!")

        st.subheader("Results table")
        #st.dataframe(df, use_container_width=True)

        # CSV ready for download
        # csv_bytes = df.to_csv(index=False).encode("utf-8")
        # st.download_button(
        #     "Download CSV",
        #     data=csv_bytes,
        #    file_name="stoma_results.csv",
        #    mime="text/csv",
        #)


if __name__ == "__main__":
    main()
