import json
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO

from config import DEFAULT_WEIGHTS_PATH
from inference import image_bytes_to_pil, process_yolo_result
from visualization import CLASS_COLORS, render_countable_area_overlay, render_detection_overlay


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def _find_weights_path(user_path: str):
    path = Path(user_path).expanduser().resolve()
    if path.exists() and path.is_file():
        return path
    return None


def _run_single_inference(model, uploaded_file, conf: float, scale: float):
    image_bytes = uploaded_file.read()
    image_pil = image_bytes_to_pil(image_bytes)
    prediction = model(image_pil, conf=conf, imgsz=640, retina_masks=True, rect=False, iou=0.25)[0]
    processed = process_yolo_result(prediction, um_per_px=scale)
    processed["filename"] = uploaded_file.name
    processed["image"] = image_pil
    return processed


def _summary_row(item: dict):
    density = item["density_info"]
    row = {
        "Filename": item["filename"],
        "Stomata count": density.get("stomata_count"),
        "Density per mm²": density.get("stomatal_density_per_mm2"),
        "Countable pixels": density.get("countable_pixels"),
        "Non-countable pixels": density.get("non_countable_pixels"),
    }
    for cls_name, cls_count in sorted(item["class_counts"].items()):
        row[f"class:{cls_name}"] = cls_count
    return row


def _download_results_button(results: list[dict]):
    serializable = []
    for item in results:
        serializable.append(
            {
                "filename": item["filename"],
                "class_counts": item["class_counts"],
                "density_info": item["density_info"],
                "stomata": item["stomata"],
            }
        )

    st.download_button(
        label="Download metadata JSON",
        data=json.dumps(serializable, indent=2).encode("utf-8"),
        file_name="sammy_stoma_results.json",
        mime="application/json",
        on_click="ignore",
    )


def main():
    st.set_page_config(page_title="Sammy Stoma Visual Analyzer", layout="wide")
    st.title("Sammy Stoma Visual Analyzer")
    st.caption("Standalone app in your folder: stomata + other classes with visualization overlays.")

    with st.sidebar:
        st.subheader("Model settings")
        default_path = str(DEFAULT_WEIGHTS_PATH)
        weights_input = st.text_input("Weights path", value=default_path)
        conf_value = st.slider("Confidence threshold", 0.01, 0.99, 0.50, 0.01)
        scale_value = st.number_input(
            "Scale (μm / pixel)",
            min_value=0.0000000001,
            max_value=10.0,
            value=10 / 46,
            step=0.0001,
            format="%0.6f",
        )

    weights_path = _find_weights_path(weights_input)
    if weights_path is None:
        st.error("Weights file not found. Set a valid path in the sidebar.")
        st.stop()

    try:
        model = load_model(str(weights_path))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload microscopy images",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded_files:
        st.info("Upload one or more images to start analysis.")
        st.stop()

    if "sammy_results" not in st.session_state:
        st.session_state["sammy_results"] = []

    if st.button("Run analysis", type="primary"):
        all_results = []
        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            status.write(f"Processing {uploaded_file.name} ({index}/{len(uploaded_files)})")
            result = _run_single_inference(model, uploaded_file, conf=conf_value, scale=scale_value)
            all_results.append(result)
            progress.progress(int(index * 100 / len(uploaded_files)))

        st.session_state["sammy_results"] = all_results
        status.success("Analysis completed")

    results = st.session_state.get("sammy_results", [])
    if not results:
        st.stop()

    st.subheader("Summary")
    summary_df = pd.DataFrame([_summary_row(item) for item in results]).set_index("Filename")
    st.dataframe(summary_df, use_container_width=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.download_button(
            label="Download summary CSV",
            data=summary_df.to_csv(index=True).encode("utf-8"),
            file_name="sammy_summary.csv",
            mime="text/csv",
            on_click="ignore",
        )
    with col_b:
        _download_results_button(results)

    st.subheader("Per-image visualization")
    st.markdown("**Class color legend:**")
    for cls_name, rgb in CLASS_COLORS.items():
        st.markdown(f"- {cls_name}: rgb{rgb}")

    for item in results:
        with st.expander(item["filename"], expanded=False):
            overlay = render_detection_overlay(item["image"], item["instances"])
            countable_overlay = render_countable_area_overlay(item["image"], item["countable_mask"])

            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1:
                st.image(item["image"], caption="Original", use_container_width=True)
            with img_col2:
                st.image(overlay, caption="Detections + masks", use_container_width=True)
            with img_col3:
                st.image(countable_overlay, caption="Countable area", use_container_width=True)

            met_col1, met_col2 = st.columns(2)
            with met_col1:
                st.markdown("**Class counts**")
                st.json(item["class_counts"])
            with met_col2:
                st.markdown("**Density info**")
                st.json(item["density_info"])

            st.markdown("**Stomata metadata**")
            st.dataframe(pd.DataFrame(item["stomata"]), use_container_width=True)


if __name__ == "__main__":
    main()
