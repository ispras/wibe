#!/bin/bash

pip install -e ./submodules/trustmark/python


# Create an array to hold pip arguments
req_args=()

# Search for all requirements.txt files in the specified directories
for dir in src/imgmarkbench/algorithms src/imgmarkbench/attacks src/imgmarkbench/datasets src/imgmarkbench/metrics; do
  echo "Searching for dependencies in directory: $dir"
  while IFS= read -r req_file; do
    echo "Found: $req_file"
    req_args+=("-r" "$req_file")
  done < <(find "$dir" -type f -name "requirements.txt")
done

# Check if any requirements files were found
if [ ${#req_args[@]} -eq 0 ]; then
  echo "No requirements.txt files found"
  exit 1
fi

# Call pip once with all -r arguments
echo "Installing all dependencies in a single pip call..."
pip install "${req_args[@]}"