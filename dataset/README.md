# Grid Image Organizer

A Python script that organizes geotagged images into a grid system based on their GPS coordinates.

## Overview

This tool takes images named with their GPS coordinates (e.g., `38.123456,-122.654321.jpg`) and sorts them into a grid of directories based on their geographic location. Images that fall outside the defined boundaries are placed in an 'unfit' directory.

## How It Works

### 1. Grid Generation
The script divides a geographic area (defined by top-left and bottom-right coordinates) into a customizable grid:
- Default: 7x7 grid covering the continental United States
- Each grid cell is named `grid-ROW-COL` (e.g., `grid-0-0`, `grid-3-7`)
- Grid cells are numbered from top-left (0-0) to bottom-right

### 2. Image Organization Process

The script follows this workflow:

1. **Reset Phase** (`move_images_to_grid_0_0()`):
   - Collects all images from existing `grid-*` directories
   - Moves all images to `grid-0-0` directory
   - Deletes empty grid directories (except `grid-0-0`)
   - Also processes any images in the 'unfit' directory

2. **Distribution Phase** (`move_images_to_their_grid()`):
   - Creates all necessary grid directories
   - Reads each image filename to extract GPS coordinates
   - Determines which grid cell contains those coordinates
   - Moves the image to the appropriate grid directory
   - Images outside all grid boundaries go to 'unfit' directory

3. **Reporting Phase** (`count_files_in_grid_folders()`):
   - Counts images in each grid directory
   - Displays a formatted breakdown of file distribution
   - Shows total count across all directories

## File Structure

```
./united_states/
├── grid-0-0/       # Top-left grid cell
├── grid-0-1/
├── grid-0-2/
├── ...
├── grid-9-9/       # Bottom-right grid cell
└── unfit/          # Images outside grid boundaries
```

## Configuration

### Default Settings
```python
# Geographic boundaries (Continental US)
top_left = (49.049081, -125.450687)      # Northwest corner
bottom_right = (24.455005, -67.343249)   # Southeast corner

# Grid dimensions
rows = 10
cols = 10

# Base directory
base_dir = "./united_states"
```

### Image Naming Convention
Images must be named with their coordinates in this format:
- `LATITUDE,LONGITUDE.jpg`
- Example: `38.897957,-77.036560.jpg` (Washington, DC)

## Usage

### Basic Usage
make sure to change directory into ./dataset before running and place the united_states folder (holding grid-0-0) in this dataset directory (same level as this README.md). Otherwise, change the base_dir in the `__main__` section.

### Custom Configuration
Modify the parameters in the `__main__` section:

```python
# Custom boundaries (e.g., for California)
top_left = (42.0, -124.5)
bottom_right = (32.5, -114.0)

# Different grid size
rows = 5
cols = 5

# Different directory
base_dir = "./california_images"
```

## Functions

### Core Functions

- **`generate_grids()`**: Creates grid boundaries and directories
- **`move_images_to_grid_0_0()`**: Consolidates all images into grid-0-0
- **`move_images_to_their_grid()`**: Distributes images to appropriate grids
- **`count_files_in_grid_folders()`**: Reports file distribution

### Debug Mode

Enable debug mode for detailed processing information:
```python
move_images_to_their_grid(top_left, bottom_right, rows, cols, base_dir, debug=True)
```

## Example Output

```
[move_images_to_grid_0_0] moved the images to grid-0-0 directory and cleaned up old grid directories
[generate_grids] Generated a 10x10 grid dictionary and created directories.
[move_images_to_their_grid] Moved 23456 images to their grid.
[move_images_to_their_grid] Moved 127 images to unfit directory.

==================================================
FILE COUNT BREAKDOWN BY GRID
==================================================
Grid       Files     
--------------------
0-0        498       
0-1        230       
...
9-9        145       
unfit      127       
--------------------
TOTAL      23583     
==================================================
```

## Edge Cases Handled

- **Alaska/Hawaii**: Images outside continental US bounds go to 'unfit'
- **Missing directories**: Automatically created as needed
- **Invalid coordinates**: Handled gracefully, placed in 'unfit'
- **Empty grids**: Shown in breakdown with count of 0

## Requirements

- Python 3.6+
- Standard library modules: `os`, `shutil`, `time`

## Notes

- The script assumes all `.jpg` files in the directories are geotagged images
- Grid boundaries are inclusive on the top and left edges, exclusive on bottom and right
- The 'unfit' directory preserves images that don't match any grid for manual review