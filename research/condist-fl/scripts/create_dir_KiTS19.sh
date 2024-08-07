#!/bin/bash

# Define source and destination folders
src_folder="/data2/open_dataset/KiTS19"
dst_folder="/neodata/open_dataset/KiTS19"

# Create the destination folder if it doesn't exist
mkdir -p "$dst_folder"

# Iterate through each subfolder
for case_dir in "$src_folder"/case_*; do
  # Get the name of the subfolder
  case_name=$(basename "$case_dir")

  # Extract the number and remove leading zeros
  case_number=$(echo "$case_name" | sed 's/case_0*//')

  # Exclude case_00210 to case_00299
  if (( case_number >= 210 && case_number <= 299 )); then
    continue
  fi

  # Define new file names
  new_imaging_file="IM_KITS_$case_number.nii.gz"
  new_segmentation_file="LB_KITS_$case_number.nii.gz"

  # Copy and rename files
  cp "$case_dir/imaging.nii.gz" "$dst_folder/$new_imaging_file"
  cp "$case_dir/segmentation.nii.gz" "$dst_folder/$new_segmentation_file"
done

echo "Done"
