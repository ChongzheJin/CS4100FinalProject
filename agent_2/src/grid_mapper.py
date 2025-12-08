import math

# Define U.S. bounding box and grid dimensions
TOP_LEFT = (49.049081, -125.450687)
BOTTOM_RIGHT = (24.455005, -67.343249)
ROWS, COLS = 7, 7  

class GridMapper:
    def __init__(self, rows=7, cols=7):
        self.rows = rows
        self.cols = cols
        self.top_left = TOP_LEFT
        self.bottom_right = BOTTOM_RIGHT
        self.num_classes = rows * cols
        self.num_classes = rows * cols
    
    def latlon_to_grid(self, lat: float, lon: float) -> int:
        """Return the grid index (0-based, row-major) for a given lat/lon."""
        top_lat, left_lon = self.top_left
        bottom_lat, right_lon = self.bottom_right

        if not (bottom_lat <= lat <= top_lat and left_lon <= lon <= right_lon):
            return -1  # out of bounds / unfit

        lat_step = (top_lat - bottom_lat) / self.rows
        lon_step = (right_lon - left_lon) / self.cols

        row = int((top_lat - lat) / lat_step)
        col = int((lon - left_lon) / lon_step)

        # Clamp to valid range
        row = min(max(row, 0), self.rows - 1)
        col = min(max(col, 0), self.cols - 1)

        return row * self.cols + col
    
    def get_grid_center(self, grid_idx: int) -> tuple:
        """Return the center lat/lon for a given grid index."""
        if grid_idx < 0 or grid_idx >= self.num_classes:
            raise ValueError(f"Grid index {grid_idx} out of range [0, {self.num_classes})")
        
        top_lat, left_lon = self.top_left
        bottom_lat, right_lon = self.bottom_right
        
        lat_step = (top_lat - bottom_lat) / self.rows
        lon_step = (right_lon - left_lon) / self.cols
        
        row = grid_idx // self.cols
        col = grid_idx % self.cols
        
        # Calculate center of the grid cell
        center_lat = top_lat - (row + 0.5) * lat_step
        center_lon = left_lon + (col + 0.5) * lon_step
        
        return center_lat, center_lon
    
    def grid_to_bounds(self, grid_idx: int) -> tuple:
        """Return the bounding box (top_left, bottom_right) for a given grid index."""
        if grid_idx < 0 or grid_idx >= self.num_classes:
            raise ValueError(f"Grid index {grid_idx} out of range [0, {self.num_classes})")
        
        top_lat, left_lon = self.top_left
        bottom_lat, right_lon = self.bottom_right
        
        lat_step = (top_lat - bottom_lat) / self.rows
        lon_step = (right_lon - left_lon) / self.cols
        
        row = grid_idx // self.cols
        col = grid_idx % self.cols
        
        # Calculate bounds of the grid cell
        cell_top = top_lat - row * lat_step
        cell_bottom = top_lat - (row + 1) * lat_step
        cell_left = left_lon + col * lon_step
        cell_right = left_lon + (col + 1) * lon_step
        
        return (cell_top, cell_left), (cell_bottom, cell_right)

# Keep the standalone function for backward compatibility
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