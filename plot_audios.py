import os
import matplotlib.pyplot as plt
from pathlib import Path

folder = "../Birds_Balanced" 

# Count audio files in each subdirectory
audio_counts = {}
for subdir in os.listdir(folder):
    subdir_path = os.path.join(folder, subdir)
    if os.path.isdir(subdir_path):
        audio_files = [f for f in os.listdir(subdir_path) if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))]
        audio_counts[subdir] = len(audio_files)

# Sort by subdirectory name
sorted_counts = sorted(audio_counts.items())
subdirs = [item[0] for item in sorted_counts]
counts = [item[1] for item in sorted_counts]

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(subdirs, counts, color='steelblue')
plt.xlabel('Subdirectory', fontsize=12)
plt.ylabel('Number of Audio Files', fontsize=12)
plt.title(f'Number of Audio Files per Subdirectory in {folder}', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)

# Add total count
total = sum(counts)
plt.text(0.5, 0.95, f'Total: {total} audio files',
         transform=plt.gca().transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('audio_count_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Total audio files: {total}")
print("\nBreakdown by subdirectory:")
for subdir, count in sorted_counts:
    print(f"  {subdir}: {count} files")
