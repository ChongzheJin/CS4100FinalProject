import math



#The purpose of this file is to turn coordinates into a single grid index value so that the 
# probability distribution can be calculated more easily


# Define U.S. bounding box and grid dimensions
TOP_LEFT = (49.049081, -125.450687)
BOTTOM_RIGHT = (24.455005, -67.343249)
ROWS, COLS = 7, 7  



#This method converts latitude and longitude coordinates into a single grid index (integer label)
def latlon_to_grid(lat: float, lon: float) -> int:
    """Return the grid index (0-based, row-major) for a given lat/lon."""
    top_lat, left_lon = TOP_LEFT
    bottom_lat, right_lon = BOTTOM_RIGHT

    if not (bottom_lat <= lat <= top_lat and left_lon <= lon <= right_lon):
        return -1  # out of bounds / unfit

    lat_step = (top_lat - bottom_lat) / ROWS
    lon_step = (right_lon - left_lon) / COLS

    row = int((top_lat - lat) / lat_step)
    col = int((lon - left_lon) / lon_step)

    # Clamp to valid range
    row = min(max(row, 0), ROWS - 1)
    col = min(max(col, 0), COLS - 1)

    return row * COLS + col  
