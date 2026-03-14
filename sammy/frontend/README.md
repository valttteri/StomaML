# Sammy Frontend

Frontend-style Streamlit app inside `sammy/frontend`, with added visualization.

## Features

- Similar structure to the original frontend (`main.py`, `config.py`, `utils.py`, `requirements.txt`, `Dockerfile`, `entrypoint.sh`)
- Local YOLO inference using project weights
- Class identification summary (`stomata`, `trichome`, `vein`, `nothing`, others)
- Visualization per image:
  - Class mask + bounding box overlays
  - Countable area overlay
  - Density + metadata tables

## Local run

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Docker run

```bash
docker build -t sammy-frontend .
docker run --rm -p 8050:8050 sammy-frontend
```

If needed, set a custom weights path with `WEIGHTS_PATH`.
