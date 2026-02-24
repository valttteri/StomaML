#!/bin/sh
set -eu

mkdir -p /tmp/.streamlit
cat > /tmp/.streamlit/config.toml <<'EOF'
[browser]
gatherUsageStats = false
EOF

export HOME=/tmp
export STREAMLIT_CONFIG_DIR=/tmp/.streamlit

exec streamlit run main.py --server.address 0.0.0.0 --server.port "${PORT:-8050}"
