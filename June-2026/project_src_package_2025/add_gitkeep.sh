#!/bin/bash

root="data_output"

if [ ! -d "$root" ]; then
  echo "Directory '$root' not found."
  exit 1
fi

# Find all subdirectories and add .gitkeep
find "$root" -type d | while read dir; do
  if [ ! -f "$dir/.gitkeep" ]; then
    touch "$dir/.gitkeep"
    echo "Created .gitkeep in $dir"
  fi
done

# chmod +x add_gitkeep.sh
# ./add_gitkeep.sh