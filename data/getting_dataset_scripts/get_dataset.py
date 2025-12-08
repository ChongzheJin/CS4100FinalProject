import random
import time
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables from the .env.local file in the ../web directory
dotenv_path = os.path.join(os.path.dirname(__file__), '../web/.env.local')
load_dotenv(dotenv_path)

# Retrieve the API key from the environment variable
api_key = os.getenv('REACT_APP_GOOGLE_MAPS_API_KEY')

# Function to generate grids (10 columns, 7 rows grid)
def generate_grids(top_left, bottom_right, col=10, row=7):
    top_left_lat, top_left_lon = top_left
    bottom_right_lat, bottom_right_lon = bottom_right

    lat_diff = top_left_lat - bottom_right_lat
    lon_diff = top_left_lon - bottom_right_lon

    lat_diff_per_grid = abs(lat_diff / row)
    lon_diff_per_grid = abs(lon_diff / col)

    return_grid = {}

    # alaska - top left (71.633799, -167.538186) - bottom right (57.589583, -141.216646)
    # hawaii - top left (22.384843, -160.323120) - bottom right (18.895523, -154.830471)
    folder_name = os.path.join("./", "grid-alaska")
    folder = [(filename.removesuffix(".jpg")) for filename in os.listdir(folder_name) if filename.endswith(".jpg")]
    return_grid["alaska"] = {
        "top_left_corner": (71.633799, -167.538186),
        "bottom_right_corner": (57.589583, -141.216646),
        "coords": folder
    }

    folder_name = os.path.join("./", "grid-hawaii")
    folder = [(filename.removesuffix(".jpg")) for filename in os.listdir(folder_name) if filename.endswith(".jpg")]
    return_grid["hawaii"] = {
        "top_left_corner": (22.384843, -160.323120),
        "bottom_right_corner": (18.895523, -154.830471),
        "coords": folder
    }

    for i in range(7):
        for j in range(10):
            folder_name = os.path.join("./", f"grid-{i}-{j}")
            folder = [(filename.removesuffix(".jpg")) for filename in os.listdir(folder_name) if filename.endswith(".jpg")]
            return_grid[f"{i}-{j}"] = {
                "top_left_corner": (top_left_lat - lat_diff_per_grid * i, top_left_lon + lon_diff_per_grid * j),
                "bottom_right_corner": (top_left_lat - lat_diff_per_grid * (i + 1), top_left_lon + lon_diff_per_grid * (j + 1)),
                "coords": folder
            }

    return return_grid

# Function to generate random coordinates within the grid
def generate_coords(top_left, bottom_right):
    lat = random.uniform(bottom_right[0], top_left[0])
    lon = random.uniform(top_left[1], bottom_right[1])
    return lat, lon

# Function to get metadata (check if location exists)
async def get_metadata(session, coords):
    metadata_URL = f"https://maps.googleapis.com/maps/api/streetview/metadata?key={api_key}&source=outdoor&location={coords[0]},{coords[1]}"
    async with session.get(metadata_URL) as response:
        metadata = await response.json()
        return metadata["status"] == "OK"

# Function to get and save street view image
async def get_streetview_image(session, coords, grid_id, base_dir = "./"):
    streetview_URL = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&location={coords[0]},{coords[1]}&source=outdoor&key={api_key}"
    
    # Send request for the image
    async with session.get(streetview_URL) as response:
        if response.status == 200:
            img_data = await response.read()
            
            # Use latitude, longitude as filename (remove any characters that might cause issues)
            lat, lon = coords
            filename = f"{lat},{lon}.jpg"
            filepath = os.path.join(base_dir, f"grid-{grid_id}", filename)

            # Check if the image already exists, and skip if it does
            if os.path.exists(filepath):
                print(f"Image for {coords} already exists: Skipping.")
                return
            
            # Save the image as a JPG file
            with open(filepath, "wb") as f:
                f.write(img_data)
            print(f"Saved image for {grid_id} | {coords}: {filename}")
        else:
            print(f"Failed to get image for {coords}")

# Process each grid to gather valid coordinates and save images
async def process_grid(session, grid_id, top_left, bottom_right, limit_per_grid, grid_coords, recorded_locations, base_dir="./"):
    grid_folder = os.path.join("./", f"grid-{grid_id}")
    grid_folder_length = len(os.listdir(grid_folder))
    valid_coords_count = grid_folder_length
    invalid_coords_count = 0

    while valid_coords_count < limit_per_grid:

        generated_coords = generate_coords(top_left, bottom_right)

        if generated_coords in recorded_locations:
            continue

        # Check if it's a valid location
        if invalid_coords_count > 10000:
            print(f"Skipping grid {grid_id} due to too many invalid locations.")
            return  # Skip this grid entirely

        if await get_metadata(session, generated_coords):
            valid_coords_count += 1
            invalid_coords_count = 0
            recorded_locations.add(generated_coords)
            
            # Save the street view image (always to grid-0-0)
            await get_streetview_image(session, generated_coords, grid_id)

            # Delay between requests to avoid API rate limits
            await asyncio.sleep(0.5)  # 0.5 second delay
            
            grid_coords[f"{grid_id}"]["coords"].append(generated_coords)
        else:
            invalid_coords_count += 1

# Main function for managing the async tasks
async def main():
    top_left = (49.049081, -125.450687)
    bottom_right = (24.455005, -67.343249)
    grid_col = 10 # number of columns
    grid_row = 7 # number of rows

    recorded_locations = set()  # Use a set to track unique locations
    grid_coords = generate_grids(top_left, bottom_right, grid_col, grid_row)

    limit_per_grid = 225 # change this to set how many images you want per grid

    # Use async session to manage all API requests
    async with aiohttp.ClientSession() as session:
        tasks = []
        for grid_id in grid_coords:
            grid_folder = os.path.join("./", f"grid-{grid_id}")
            grid_folder_length = len(os.listdir(grid_folder))
            if grid_folder_length < limit_per_grid and grid_folder_length != 0:
                task = asyncio.create_task(process_grid(session, 
                                                        grid_id, 
                                                        grid_coords[grid_id]["top_left_corner"], 
                                                        grid_coords[grid_id]["bottom_right_corner"], 
                                                        limit_per_grid, 
                                                        grid_coords, 
                                                        recorded_locations))
                tasks.append(task)
            else:
                print("Quota filled for grid ", grid_id)

        # Wait for all tasks to finish
        await asyncio.gather(*tasks)

    # Optionally, print the results
    for grid_id, grid_data in grid_coords.items():
        print(f"Grid {grid_id} - {len(grid_data['coords'])} valid locations found.")
    
if __name__ == "__main__":
    asyncio.run(main())
