from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO

from config import WEIGHTS_PATH
from inference import image_bytes_to_pil, process_yolo_result
from utils import ProgressBar, RemainingTime, download_csv, download_metadata
from visualization import CLASS_COLORS, render_countable_area_overlay, render_detection_overlay

MAX_VIS_IMAGES = 10

_CSS = """
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 0% 0%, rgba(46, 125, 82, 0.08), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(30, 42, 56, 0.12), transparent 55%),
        linear-gradient(180deg, #f3f6fb 0%, #edf2f8 100%);
}
[data-testid="stSidebar"] {
    background: #1e2a38;
}
[data-testid="stSidebar"] * {
    color: #d6e4f0 !important;
}
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #a8c4d8 !important;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Hero header ── */
.bsa-hero {
    background: linear-gradient(128deg, #1e2a38 0%, #2a3d50 45%, #2e7d52 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 10px 30px rgba(30, 42, 56, 0.18);
}
.bsa-hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.bsa-hero p {
    font-size: 0.95rem;
    opacity: 0.78;
    margin: 0;
}

/* ── Section headings ── */
.bsa-section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e2a38;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-left: 4px solid #2e7d52;
    padding-left: 0.65rem;
    margin: 1.8rem 0 0.9rem;
}

/* ── Cards ── */
.bsa-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

/* ── Legend chips ── */
.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 0.28rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #2d3748;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ── Upload area tweak ── */
[data-testid="stFileUploader"] {
    background: #fff;
    border-radius: 10px;
    padding: 0.5rem;
    border: 1px solid #dbe4ef;
}

/* ── Button accent ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #2e7d52;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.55rem 1.6rem;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #25663f;
}

/* ── Expander header clarity ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stExpander"] details summary {
    background: #f8fbff;
    border-radius: 10px;
    padding: 0.25rem 0.4rem;
}
[data-testid="stExpander"] details summary p {
    color: #132236 !important;
    font-weight: 700;
    letter-spacing: 0.01em;
}

/* ── Dataframe ── */
[data-testid="stDataFrameResizable"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def _find_weights_path(path_str: str):
    path = Path(path_str).expanduser().resolve()
    if path.exists() and path.is_file():
        return path
    return None


def _run_single_image(model, uploaded_file, conf: float, scale: float):
    image_bytes = uploaded_file.getvalue()
    image_pil = image_bytes_to_pil(image_bytes)
    result = model(image_pil, conf=conf, imgsz=640, retina_masks=True, rect=False, iou=0.25)[0]
    processed = process_yolo_result(result, um_per_px=scale)
    processed["filename"] = uploaded_file.name
    processed["image"] = image_pil
    return processed


def _build_summary_df(results: list[dict]):
    rows = []
    for item in results:
        density = item["density_info"]
        row = {
            "Filename": item["filename"],
            "stomata_count": density.get("stomata_count"),
            "stomatal_density_per_mm2": density.get("stomatal_density_per_mm2"),
            "countable_pixels": density.get("countable_pixels"),
            "non_countable_pixels": density.get("non_countable_pixels"),
        }
        for class_name, class_count in sorted(item["class_counts"].items()):
            row[f"class:{class_name}"] = class_count
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Filename")


def _legend_html():
    chips = ""
    for cls_name, (r, g, b) in CLASS_COLORS.items():
        chips += (
            f'<span class="legend-chip">'
            f'<span class="legend-swatch" style="background:rgb({r},{g},{b})"></span>'
            f'{cls_name}'
            f'</span>'
        )
    return f'<div class="legend-row">{chips}</div>'


def main():
    st.set_page_config(
        page_title="Birch Stomata Analysis and Visualization",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="bsa-hero">'
        '<h1>Birch Stomata Analysis and Visualization</h1>'
        '<p>Upload microscopy images to detect and measure stomata, trichomes, veins, '
        'and other structures using a YOLOv8 segmentation model.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if "inference_done" not in st.session_state:
        st.session_state["inference_done"] = False
    if "sammy_frontend_results" not in st.session_state:
        st.session_state["sammy_frontend_results"] = []
    if "file_uploader" not in st.session_state:
        st.session_state["file_uploader"] = 1

    # ── Sidebar: model settings ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")
        st.divider()
        weights_input = st.text_input(
            "Weights path",
            value=WEIGHTS_PATH,
            help="Absolute or relative path to the YOLO weights (.pt) file.",
        )
        st.markdown("---")
        scale_value = st.number_input(
            label="Image scale (μm / pixel)",
            min_value=0.0000000001,
            max_value=10.0,
            value=10 / 46,
            step=0.0001,
            format="%0.6f",
            help="Default: 10 μm / 46 px ≈ 0.2174",
        )
        conf_value = st.number_input(
            label="Confidence threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            format="%0.2f",
            help="Minimum detection confidence (0.01 – 0.99).",
        )
        st.markdown("---")
        st.caption("Visualization is limited to the first 10 images.")

    # ── Validate model ────────────────────────────────────────────────────────
    weights_path = _find_weights_path(weights_input)
    if weights_path is None:
        st.error("⚠️ Weights file not found. Update the path in the sidebar.")
        st.stop()
    try:
        model = load_model(str(weights_path))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<p class="bsa-section-heading">Upload Images</p>', unsafe_allow_html=True)

    upload_col, meta_col = st.columns([2, 1], gap="large")
    with upload_col:
        uploaded_files = st.file_uploader(
            "Drop JPG / PNG microscopy images here",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
            key=st.session_state["file_uploader"],
            label_visibility="collapsed",
        )
    with meta_col:
        st.metric(label="Files queued", value=len(uploaded_files))
        if len(uploaded_files) > 0 and st.button(label="Clear files", icon=":material/delete:"):
            st.session_state["file_uploader"] += 1
            st.session_state["inference_done"] = False
            st.session_state["sammy_frontend_results"] = []
            st.rerun()

    st.markdown("")
    run_btn = st.button("▶  Run Analysis", type="primary", disabled=not uploaded_files)

    # ── Run inference ─────────────────────────────────────────────────────────
    if run_btn and uploaded_files:
        results = []
        st.session_state["inference_done"] = False

        time_approximator = RemainingTime(total_files=len(uploaded_files))
        status = st.empty()
        status.markdown(
            '<div class="bsa-card"><strong>Analysis in progress</strong> &mdash; Time remaining: …</div>',
            unsafe_allow_html=True,
        )
        progress = ProgressBar(0, "stretch")
        progress.render()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            start = datetime.now()
            item = _run_single_image(model, uploaded_file, conf=conf_value, scale=scale_value)
            end = datetime.now()
            results.append(item)
            ratio = int(round(idx / len(uploaded_files), 2) * 100)
            progress.update(ratio)
            time_left = time_approximator.get_update(end - start)
            status.markdown(
                f'<div class="bsa-card"><strong>Analysing {idx} / {len(uploaded_files)}</strong>'
                f' &mdash; Time remaining: {time_left}</div>',
                unsafe_allow_html=True,
            )

        progress.update(100)
        status.success(f"Analysis complete — {len(results)} image(s) processed.")
        st.session_state["sammy_frontend_results"] = results
        st.session_state["inference_done"] = True

    if not st.session_state["inference_done"]:
        st.stop()

    results = st.session_state["sammy_frontend_results"]
    if not results:
        st.info("No results available.")
        st.stop()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<p class="bsa-section-heading">Summary Results</p>', unsafe_allow_html=True)

    summary_df = _build_summary_df(results)
    st.dataframe(summary_df, use_container_width=True)

    dl_col1, dl_col2 = st.columns([1, 1], gap="small")
    with dl_col1:
        download_csv(summary_df)
    with dl_col2:
        download_metadata(results)

    # ── Visualization ─────────────────────────────────────────────────────────
    st.markdown('<p class="bsa-section-heading">Image Visualization</p>', unsafe_allow_html=True)

    if len(results) > MAX_VIS_IMAGES:
        st.info(
            f"Visualization is shown for the first **{MAX_VIS_IMAGES}** of **{len(results)}** images. "
            "Use the downloads above to access full results."
        )

    st.markdown("**Detection class legend**")
    st.markdown(_legend_html(), unsafe_allow_html=True)

    for item in results[:MAX_VIS_IMAGES]:
        with st.expander(f"Image: {item['filename']}", expanded=False):
            overlay = render_detection_overlay(item["image"], item["instances"])
            countable = render_countable_area_overlay(item["image"], item["countable_mask"])

            img_col1, img_col2, img_col3 = st.columns(3, gap="small")
            with img_col1:
                st.image(item["image"], caption="Original", use_container_width=True)
            with img_col2:
                st.image(overlay, caption="Class mask overlay", use_container_width=True)
            with img_col3:
                st.image(countable, caption="Countable area overlay", use_container_width=True)

            st.markdown("")
            info_col1, info_col2 = st.columns(2, gap="medium")
            with info_col1:
                st.markdown("**Class counts**")
                st.json(item["class_counts"])
            with info_col2:
                st.markdown("**Density info**")
                st.json(item["density_info"])

            st.markdown("**Per-stomata metadata**")
            st.dataframe(pd.DataFrame(item["stomata"]), use_container_width=True)


if __name__ == "__main__":
    main()
