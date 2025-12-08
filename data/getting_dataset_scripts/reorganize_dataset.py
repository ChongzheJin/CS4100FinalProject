import os
import shutil
import time

def generate_grids(top_left, bottom_right, rows=9, cols=10, base_dir="./", debug=False):
    """
    Generates coordinate bounds for the grid (top left, bottom right).
    Also creates grid directories if they don't exist.
    """
    
    # initialize starting points
    top_left_lat, top_left_lon = top_left
    bottom_right_lat, bottom_right_lon = bottom_right

    # size of the total map bound we're working with
    lat_diff = top_left_lat - bottom_right_lat
    lon_diff = top_left_lon - bottom_right_lon

    # coordinate distance for each grid (from the top left point)
    lat_diff_per_grid = abs(lat_diff / rows)
    lon_diff_per_grid = abs(lon_diff / cols)

    # generate a dictionary with the grid bound and initialize an array that holds all coordinates that belongs to that grid
    # each grid is named 'row-col'
    return_grid = {}
    for i in range(rows):
        for j in range(cols):
            grid_name = f"grid-{i}-{j}"
            grid_path = os.path.join(base_dir, grid_name)
            
            # Create the directory if it doesn't exist
            if not os.path.exists(grid_path):
                os.makedirs(grid_path)
                if debug:
                    print(f"Created directory: {grid_path}")
            
            return_grid[f"{i}-{j}"] = {
                "top_left_corner": (top_left_lat - lat_diff_per_grid * i, top_left_lon + lon_diff_per_grid * j),
                "bottom_right_corner": (top_left_lat - lat_diff_per_grid * (i + 1), top_left_lon + lon_diff_per_grid * (j + 1)),
                "coords": []
            }

    # debug information for each grid bound
    if debug:
        for i in range(rows):
            for j in range(cols):
                print(f"grid {i}-{j}: ")
                print("  top left    : ", return_grid[f"{i}-{j}"]["top_left_corner"])
                print("  bottom right: ", return_grid[f"{i}-{j}"]["bottom_right_corner"])
    
    print(f"[generate_grids] Generated a {rows}x{cols} grid dictionary and created directories.")
    return return_grid

def move_images_to_their_grid(top_left, bottom_right, rows=9, cols=10, base_dir="./", debug=False):
    '''
    Moves the image files from the base directory grid-0-0 to each grid that fits each grid boundary.
    Images that don't fit in any grid are moved to an 'unfit' directory.
    '''
    grid = generate_grids(top_left, bottom_right, rows, cols, base_dir, debug)

    grid_0_0_folder = os.path.join(base_dir, "grid-0-0")
    unfit_folder = os.path.join(base_dir, "unfit")
    
    # Create unfit directory if it doesn't exist
    if not os.path.exists(unfit_folder):
        os.makedirs(unfit_folder)
        if debug:
            print(f"Created unfit directory: {unfit_folder}")
    
    total_files_moved = 0
    unfit_files = 0

    # Process all files in grid-0-0
    for file in os.listdir(grid_0_0_folder):
        if not file.endswith(".jpg"):
            continue
            
        coords = file.removesuffix(".jpg").split(",")
        lat = float(coords[0])
        long = float(coords[1])
        
        file_placed = False
        
        # Try to place file in appropriate grid
        for i in range(rows):
            for j in range(cols):
                grid_name = f"grid-{i}-{j}"
                grid_num = f"{i}-{j}"
                
                between_lat = grid[grid_num]["bottom_right_corner"][0] <= lat < grid[grid_num]["top_left_corner"][0]
                between_long = grid[grid_num]["top_left_corner"][1] <= long < grid[grid_num]["bottom_right_corner"][1]
                
                if between_lat and between_long:
                    grid[grid_num]["coords"].append((lat, long))
                    file_path = os.path.join(grid_0_0_folder, file)
                    new_path = os.path.join(base_dir, grid_name, file)
                    shutil.move(file_path, new_path)
                    total_files_moved += 1
                    file_placed = True
                    break
            
            if file_placed:
                break
        
        # If file doesn't fit in any grid, move to unfit directory
        if not file_placed:
            file_path = os.path.join(grid_0_0_folder, file)
            new_path = os.path.join(unfit_folder, file)
            shutil.move(file_path, new_path)
            unfit_files += 1
            if debug:
                print(f"Moved unfit file: {file} (lat: {lat}, long: {long})")
    
    # debug info for grid boundaries
    if debug:
        print(f"grid alaska: ")
        print("  top left    : ", grid["alaska"]["top_left_corner"])
        print("  bottom right: ", grid["alaska"]["bottom_right_corner"])
        print("        coords: ", grid["alaska"]["coords"])
        print(f"grid hawaii: ")
        print("  top left    : ", grid["hawaii"]["top_left_corner"])
        print("  bottom right: ", grid["hawaii"]["bottom_right_corner"])
        print("        coords: ", grid["hawaii"]["coords"])
        for i in range(7):
            for j in range(10):
                print(f"grid {i}-{j}: ")
                print("  top left    : ", grid[f"{i}-{j}"]["top_left_corner"])
                print("  bottom right: ", grid[f"{i}-{j}"]["bottom_right_corner"])
                print("        coords: ", grid[f"{i}-{j}"]["coords"])
        print("========================")
        print(f"Total files moved to grids: {total_files_moved}")
        print(f"Total unfit files: {unfit_files}")
    
    print(f"[move_images_to_their_grid] Moved {total_files_moved} images to their grid.")
    print(f"[move_images_to_their_grid] Moved {unfit_files} images to unfit directory.")

                
# Function to move all images from directories that start with "grid" into "grid-0-0"
def move_images_to_grid_0_0(base_dir="./", debug=False):
    grid_0_0_folder = os.path.join(base_dir, "grid-0-0")
    
    # Ensure grid-0-0 exists (create parent directories if needed)
    os.makedirs(grid_0_0_folder, exist_ok=True)
    if not os.path.exists(grid_0_0_folder):
        print(f"Created {grid_0_0_folder} directory")
    
    # List to keep track of directories to delete
    dirs_to_delete = []
    
    # Iterate over all folders in the base directory
    total_files_moved = 0
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Process grid directories and unfit directory
            if folder_name.startswith("grid") or folder_name == "unfit":
                # Skip grid-0-0 itself
                if folder_name == "grid-0-0":
                    continue
                    
                # Mark directory for deletion
                dirs_to_delete.append(folder_path)
                
                # Iterate over all files in the folder
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if file.endswith(".jpg") and os.path.isfile(file_path):  # Check if it's a .jpg image
                        # Move the image to "grid-0-0"
                        new_path = os.path.join(grid_0_0_folder, file)
                        shutil.move(file_path, new_path)
                        total_files_moved += 1
    
    # Delete the now-empty grid directories and unfit directory (except grid-0-0)
    for dir_path in dirs_to_delete:
        try:
            shutil.rmtree(dir_path)
            if debug:
                print(f"Deleted directory: {dir_path}")
        except Exception as e:
            print(f"Error deleting {dir_path}: {e}")
    
    print(f"Total files moved: {total_files_moved}")
    print(f"Total directories deleted: {len(dirs_to_delete)}")
    print("[move_images_to_grid_0_0] moved the images to grid-0-0 directory and cleaned up old grid directories")


def count_files_in_grid_folders(base_dir="./", show_breakdown=True):
    """
    Count all files in grid directories and optionally show breakdown per grid.
    Also counts files in 'unfit' directory if it exists.
    """
    grid_counts = {}
    unfit_count = 0
    total_count = 0
    
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return 0
    
    # Count files in each directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            if folder_name.startswith("grid-"):
                # Count files in grid folder
                file_count = sum(1 for f in os.listdir(folder_path) 
                                if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".jpg"))
                
                # Extract grid coordinates for proper sorting
                grid_coords = folder_name.replace("grid-", "")
                grid_counts[grid_coords] = file_count
                total_count += file_count
                
            elif folder_name == "unfit":
                # Count files in unfit folder
                unfit_count = sum(1 for f in os.listdir(folder_path) 
                                if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".jpg"))
                total_count += unfit_count
    
    # Show breakdown if requested
    if show_breakdown:
        print("\n" + "="*50)
        print("FILE COUNT BREAKDOWN BY GRID")
        print("="*50)
        
        # Sort grid names properly (by row then column)
        sorted_grids = sorted(grid_counts.keys(), 
                            key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
        
        # Print in a formatted table
        print(f"{'Grid':<10} {'Files':<10}")
        print("-"*20)
        
        for grid in sorted_grids:
            count = grid_counts[grid]
            print(f"{grid:<10} {count:<10}")
            
        if unfit_count > 0:
            print(f"{'unfit':<10} {unfit_count:<10}")
        
        print("-"*20)
        print(f"{'TOTAL':<10} {total_count:<10}")
        print("="*50 + "\n")
    
    return total_count

if __name__ == "__main__":

    # initial values
    top_left = (49.049081, -125.450687) # top left bound of our map
    bottom_right = (24.455005, -67.343249) # bottom right bound for our map

    # number of rows and cols for the grid
    rows = 7
    cols = 7
    
    # base directory for all operations
    base_dir = "./united_states"

    # ===== divide and place images into grids =====

    # reset all images to grid-0-0 directory
    move_images_to_grid_0_0(base_dir)

    # move all the images from grid-0-0 out to their respective grid directory
    move_images_to_their_grid(top_left, bottom_right, rows, cols, base_dir)

    # Show file count breakdown
    count_files_in_grid_folders(base_dir, show_breakdown=True)

    # do not be startled if some of these have 0 images in their grid. It might just
    # be over the ocean