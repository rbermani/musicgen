import os
import sys

# Get the project path
PROJECT_PATH = os.getcwd()

# Path to the source directory
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")

# Add the source directory to sys.path
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)

print("sys.path:", sys.path)