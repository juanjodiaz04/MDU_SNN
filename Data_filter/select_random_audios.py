import os
import shutil
import random
from pathlib import Path

source_folder = "Birds_clean"
output_folder = "Birds_Balanced"
N_audios = 330

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each subdirectory
for subdir in os.listdir(source_folder):
    subdir_path = os.path.join(source_folder, subdir)

    if os.path.isdir(subdir_path):
        # Get all audio files
        audio_files = [f for f in os.listdir(subdir_path) if f.endswith('.ogg')]

        # Check if folder has at least N_audios files
        if len(audio_files) >= N_audios:
            # Randomly select N_audios files
            selected_files = random.sample(audio_files, N_audios)

            # Create output subfolder
            output_subdir = os.path.join(output_folder, subdir)
            os.makedirs(output_subdir, exist_ok=True)

            # Copy selected files
            for file in selected_files:
                src = os.path.join(subdir_path, file)
                dst = os.path.join(output_subdir, file)
                shutil.copy2(src, dst)

            print(f"✓ {subdir}: Selected and copied {N_audios} files (from {len(audio_files)} available)")
        else:
            print(f"✗ {subdir}: Only {len(audio_files)} files available (less than {N_audios})")

print(f"\nAll selected files saved to: {output_folder}/")
