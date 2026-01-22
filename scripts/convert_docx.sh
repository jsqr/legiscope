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

municipality_name=$1

if [ -z "$municipality_name" ]; then
    echo "Usage: $0 <municipality_name>"
    echo "Example: $0 CA-LosAngeles"
    exit 1
fi

echo "Converting docx files to plain text for $municipality_name"

# Set the input directory to be data/laws/municipality_name
input_dir="data/laws/$municipality_name/raw" 

# Set the output to go into data/laws/municipality_name/processed/code.txt
output_file="data/laws/$municipality_name/processed/code.txt"

# Use a temporary directory to store the intermediate text files
temp_dir="data/temp"
mkdir -p "$temp_dir"

# Check if the output file already exists and delete it if necessary
if [ -f "$output_file" ]; then
    echo "Output file $output_file already exists. Deleting it."
    rm "$output_file"
fi

# Iterate over all docx files in the input directory
for file in "$input_dir"/*.docx; do
    if [ -f "$file" ]; then
        # Extract the file name without the .docx extension
        filename=$(basename "$file" .docx)

        # Convert using the best available tool
        if [ "$CONVERTER" = "textutil" ]; then
            # macOS textutil
            textutil -convert txt -output "$temp_dir/$filename.txt" "$file"
        else
            # Fallback to pandoc (preserves Unicode)
            pandoc "$file" -f docx -t plain --output "$temp_dir/$filename.txt"
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
