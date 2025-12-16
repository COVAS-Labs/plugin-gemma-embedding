#!/bin/bash

# Delete dist if it already exists
if [ -d "dist" ]; then
    rm -rf dist
fi

# Create dist
mkdir dist

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install --target ./deps -r requirements.txt
fi

# Remember to add any additional files, and change the name of the plugin
artifacts=(
    "cn-plugin-gemma-embedding.py"
    "requirements.txt"
    "manifest.json" "__init__.py"
    "model/added_tokens.json"
    "model/config.json"
    "model/generation_config.json"
    "model/special_tokens_map.json"
    "model/tokenizer.json"
    "model/tokenizer.model"
    "model/tokenizer_config.json"
    "model/onnx/model_fp16.onnx"
    "model/onnx/model_fp16.onnx_data"
)

if [ -d "deps" ]; then
    artifacts+=("deps")
fi

# Create the zip archive
zip -r -9 "dist/cn-plugin-gemma-embedding.zip" "${artifacts[@]}"
