"""
Usage to monitor current folder:
python scripts/monitor_folder.py --folder_path ./path/to/checkpoint
"""

import os
import time

import fire


size_limit = 60 * 1024**3  # 100GB in bytes


def get_folder_size(folder):
    """Calculate the total size of all files in a folder."""
    total_size = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    return total_size


def remove_oldest_files(folder, target_size):
    """Remove oldest files until the folder size is below the target size."""
    # Collect files and their modification times
    files = []
    for root, dirs, files_in_dir in os.walk(folder):
        for file in files_in_dir:
            filepath = os.path.join(root, file)
            files.append((filepath, os.path.getmtime(filepath)))

    # Sort files by modification time (oldest first)
    files.sort(key=lambda x: x[1])

    # Remove files until the folder size is below the target size
    total_size = get_folder_size(folder)
    for filepath, _ in files:
        if total_size <= target_size:
            break
        total_size -= os.path.getsize(filepath)
        os.remove(filepath)
        print(f"Removed: {filepath}")


def main(folder_path, limit=size_limit):
    """Continuously monitor folder size and enforce limit."""
    print(folder_path)
    assert "checkpoint" in folder_path, "Please provide a folder path containing 'checkpoint' in the name."
    while True:
        folder_size = get_folder_size(folder_path)
        print(f"Current folder size: {folder_size / (1024**3):.2f} GB")
        if folder_size > limit:
            print("Folder size exceeds limit. Removing files...")
            remove_oldest_files(folder_path, limit)
        time.sleep(300)  # Check every 5 mins


if __name__ == "__main__":
    fire.Fire(main)
