#!/bin/bash

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "pandoc is not installed. Please install it before running this script."
    exit 1
fi

municipality_name=$1

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

        # Use pandoc to convert the docx file to plain text (still has some Unicode characters)
        #pandoc "$file" -f docx -t plain --output "$temp_dir/$filename.txt"

        # Use pandoc to convert the docx file to plain text with ASCII encoding
        pandoc "$file" -f docx -t plain --ascii --output "$temp_dir/$filename.txt"
        # Replace &nbsp; with a space, &ndash; with a hyphen, &sect; with a section symbol, etc.
        sed -i '' 's/&nbsp;/ /g' "$temp_dir/$filename.txt"
        sed -i '' 's/&ndash;/-/g' "$temp_dir/$filename.txt"
        sed -i '' 's/&sect;/§/g' "$temp_dir/$filename.txt"
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
