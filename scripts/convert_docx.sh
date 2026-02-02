#!/bin/bash

# Detect best available converter (textutil on macOS is ~100x faster than pandoc)
if command -v textutil &> /dev/null; then
    CONVERTER="textutil"
    echo "Using textutil (macOS native) for fast conversion"
elif command -v pandoc &> /dev/null; then
    CONVERTER="pandoc"
    echo "Using pandoc for conversion (slower for large files)"
else
    echo "No DOCX converter found. Install pandoc: brew install pandoc"
    exit 1
fi

# New format: expect full path to raw directory
raw_dir=$1

if [ -z "$raw_dir" ]; then
    echo "Usage: $0 <raw_directory_path>"
    echo "Example: $0 data/laws/IL/Springfield/municipal-code/raw"
    exit 1
fi

if [ ! -d "$raw_dir" ]; then
    echo "Error: Raw directory does not exist: $raw_dir"
    exit 1
fi

# Extract jurisdiction info from path for logging
# Format: data/laws/STATE/LOCALITY/CODE_SLUG/raw
path_parts=(${raw_dir//\// })
state=${path_parts[2]}
locality=${path_parts[3]}
code_slug=${path_parts[4]}

echo "Converting DOCX files for $state-$locality ($code_slug)"

# Output file goes in the code directory (parent of raw)
code_dir=$(dirname "$raw_dir")
output_file="$code_dir/code.txt"

# Use a temporary directory to store the intermediate text files
temp_dir="data/temp"
mkdir -p "$temp_dir"

# Check if the output file already exists and delete it if necessary
if [ -f "$output_file" ]; then
    echo "Output file $output_file already exists. Deleting it."
    rm "$output_file"
fi

# Iterate over all docx files in the input directory
for file in "$raw_dir"/*.docx; do
    if [ -f "$file" ]; then
        # Extract the file name without the .docx extension
        filename=$(basename "$file" .docx)

        # Convert using the best available tool
        if [ "$CONVERTER" = "textutil" ]; then
            # macOS textutil
            textutil -convert txt -output "$temp_dir/$filename.txt" "$file"
        else
            # Fallback to pandoc
            # Use ASCII encoding + restoration to ensure no hidden artifacts (like NBSP) remain
            pandoc "$file" -f docx -t plain --ascii --output "$temp_dir/$filename.txt"

            # Handle sed in-place argument for both macOS and Linux
            if [[ "$OSTYPE" == "darwin"* ]]; then
                SED_OPTS=(-i '')
            else
                SED_OPTS=(-i)
            fi

            # Restore symbols and clean artifacts
            sed "${SED_OPTS[@]}" 's/&nbsp;/ /g' "$temp_dir/$filename.txt"
            sed "${SED_OPTS[@]}" 's/&ndash;/-/g' "$temp_dir/$filename.txt"
            sed "${SED_OPTS[@]}" 's/&sect;/§/g' "$temp_dir/$filename.txt"
        fi
        echo "  Converted: $filename.docx"
    else
        echo "'$file' is not a file. Skipping..."
    fi
done

# Check if any text files were generated
if [ -n "$(ls -A "$temp_dir")" ]; then

    # Concatenate the text files into the output file with a newline between each
    for txt_file in "$temp_dir"/*.txt; do
        cat "$txt_file" >> "$output_file"
        echo "" >> "$output_file"  # Add a newline
    done

    echo "Output written to $output_file"
else
    echo "No text files were generated."
fi

echo "Conversion complete--cleaning up temporary files in $temp_dir"
# Clean up the temporary directory
rm -rf "$temp_dir"
