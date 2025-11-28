import numpy as np
import torch

from agent_2.grid_mapper import GridMapper
GRID_MAPPER = GridMapper(rows=7, cols=7)


# Computes the distance between two lat/lon points in kilometers
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c



# Computes a 49 length vector (one per grid) where closer distances get a value closer
# to 1 and farther distances get a value closer to 0
def compute_likelihood_terms(lat1, lon1, sigma_km=800.0):
    num_grids = GRID_MAPPER.num_classes
    likelihoods = np.zeros(num_grids)

    for g in range(num_grids):
        lat_g, lon_g = GRID_MAPPER.get_grid_center(g)
        dist = haversine_km(lat1, lon1, lat_g, lon_g)

        # Gaussian-like decay
        likelihoods[g] = np.exp(-(dist**2) / (2 * sigma_km**2))

    return likelihoods



# Combines both agents to output a single coord (using likelihood term and probability)
def combine_agents(img, agent1, agent2):

    # Agent 1 output coords
    with torch.no_grad():
        lat1, lon1 = agent1(img)[0].tolist()

    # Conver coords from agent one to likelihood terms
    likelihoods = compute_likelihood_terms(lat1, lon1)

    # Agent 2 output probability distributions 
    with torch.no_grad():
        logits = agent2(img)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Score
    scores = probs * likelihoods

    if scores.sum() == 0:
        scores = probs

    # Pick grid with highest score
    best_grid = int(np.argmax(scores))

    # Return center coordinate of chosen grid
    final_lat, final_lon = GRID_MAPPER.get_grid_center(best_grid)

    return final_lat, final_lon, {
        "agent1_latlon": (lat1, lon1),
        "grid_likelihood": likelihoods,
        "agent2_probs": probs,
        "combined_scores": scores,
        "chosen_grid": best_grid
    }
