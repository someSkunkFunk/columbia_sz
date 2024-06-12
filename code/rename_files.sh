#!/bin/bash

# helper script for renaming files I stupidly misnamed
# Define base directory
BASE_DIR="../results/evnt_decimate/thresh_750_5fold_shuffled"

#Find all files within base dir without an extension 
find "$BASE_DIR" -type f ! -name "*.*" | while read -r FILE; do
    # remove trailing underscore
    NEW_NAME=$(echo "$FILE" | sed 's/_$//')
    # append pkl extension
    mv "$FILE" "$NEW_NAME.pkl"
done

echo "Renaming completed"