#!/bin/sh

file_id="1MY_9yI43AsNtO2GlW5x7FeaDt06z6-JS"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    python -m pip install gdown
fi

echo "Downloading the file..."
gdown --id "$file_id"
echo "Download complete!"
