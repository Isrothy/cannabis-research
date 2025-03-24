#!/bin/bash

models=("valhalla/distilbart-mnli-12-9" "facebook/bart-large-mnli" "roberta-large-mnli")

collections=("googleApps" "appleApps")

for model in "${models[@]}"; do
    echo "Processing model: $model"
    for collection in "${collections[@]}"; do
        echo "Processing collection: $collection"
        python python/llm_feature_generator.py --collection $collection --model $model
    done
done
