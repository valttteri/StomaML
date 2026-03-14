# Sammy Stoma Visual Analyzer

Standalone Streamlit app for stomata and other class identification with visualization overlays.

## What this app adds

- Local inference with the same YOLO weights used by the existing project.
- Class-wise identification for `stomata`, `trichome`, `vein`, `nothing` (and any extra model classes).
- Visualization not present in the original app:
  - Mask + bounding box overlay with class labels and confidence.
  - Countable-area overlay.
  - Per-image class counts and stomata metadata table.

## Run

From this `sammy` folder:

```bash
pip install -r requirements.txt
streamlit run main.py
```

Default weights path points to:

`../backend/models/weights.pt`

If your weights are elsewhere, update the path in the app sidebar.
