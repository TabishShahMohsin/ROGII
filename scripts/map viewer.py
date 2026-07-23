import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# =====================================================================
# CONFIGURATION & USER SETTINGS
# =====================================================================
# Set the target well name at the top of the file to explore instantly
WELL_NAME = '1b1eba53'
TOP_K_NEIGHBORS = 3
DATA_DIR = '../data/train'

# Set global plotting aesthetics for a professional, high-performance look
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def load_well_df(well_name):
    """Loads a single well CSV on demand for high memory efficiency."""
    filepath = os.path.join(DATA_DIR, f"{well_name}__horizontal_well.csv")
    if not os.path.exists(filepath):
        matches = glob.glob(f'*{well_name}__horizontal_well.csv')
        if matches:
            filepath = matches[0]
        else:
            matches_train = glob.glob(f'../data/train/*{well_name}*__horizontal_well.csv')
            if matches_train:
                filepath = matches_train[0]
    return pd.read_csv(filepath)

def calculate_well_similarity(well_a_df, well_b_df):
    """
    Computes physical proximity and parallelism between two wells efficiently.
    Expects dataframes with columns: 'X', 'Y', 'Z' (or TVD)
    """
    coords_a = well_a_df[['X', 'Y', 'Z']].values
    coords_b = well_b_df[['X', 'Y', 'Z']].values
    
    # Calculate physical distance (Proximity via KDTree / cdist)
    distances_a_to_b = cdist(coords_a, coords_b, metric='euclidean')
    min_distances = np.min(distances_a_to_b, axis=1)
    mean_physical_distance = np.mean(min_distances)

    vec_a = coords_a[-1] - coords_a[0]
    vec_b = coords_b[-1] - coords_b[0]
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        parallelism = 0.0
    else:
        unit_a = vec_a / norm_a
        unit_b = vec_b / norm_b
        parallelism = float(np.dot(unit_a, unit_b))
        
    return {
        "distance_ft": float(mean_physical_distance),
        "parallelism": parallelism,
    }

def safe_rmse(y1, y2):
    """Safely computes Root Mean Square Error between two arrays, ignoring NaNs."""
    valid = ~np.isnan(y1) & ~np.isnan(y2)
    if not np.any(valid):
        return 999.0
    return float(np.sqrt(np.mean((y1[valid] - y2[valid])**2)))

def project_neighbor_tvt(w1_df, w2_df):
    """
    Maps the known TVT + Z from neighbor W2 onto target W1 
    by finding the closest points in 3D (X, Y, Z) space.
    """
    w2_coords = w2_df[['X', 'Y', 'Z']].values
    w1_coords = w1_df[['X', 'Y', 'Z']].values
    
    tree = KDTree(w2_coords)
    distances, closest_w2_indices = tree.query(w1_coords)
    
    w1_mapped = w1_df.copy()
    w1_mapped['n_ztvt'] = (w2_df['TVT'] + w2_df['Z']).iloc[closest_w2_indices].values
    w1_mapped['dist_n'] = distances
    
    return w1_mapped

print(f"Scanning well datasets from {DATA_DIR}...")
csv_files = glob.glob(os.path.join(DATA_DIR, '*__horizontal_well.csv'))
if not csv_files:
    csv_files = glob.glob('*__horizontal_well.csv')

well_names = []
start_points = []
ms = []

for filepath in sorted(csv_files):
    filename = os.path.basename(filepath)
    w_name = filename.split('__')[0]
    try:
        # Read only necessary columns for fast metadata indexing
        df = pd.read_csv(filepath, usecols=['X', 'Y', 'Z', 'MD', 'TVT_input'])
        evalz = df[df['TVT_input'].isna()]
        if len(evalz) == 0:
            evalz = df
        try:
            m, c = np.polyfit(evalz['MD'], evalz['Z'], deg=1)
        except:
            m = 0.0
        start_points.append((evalz['X'].iloc[0], evalz['Y'].iloc[0]))
        ms.append(m)
        well_names.append(w_name)
    except Exception as e:
        pass

print(f"Successfully indexed {len(well_names)} wells.")

start_points = np.array(start_points)
start_tree = KDTree(start_points)

# Select target well based on top configuration variable
if WELL_NAME in well_names:
    current_well = WELL_NAME
else:
    print(f"Well '{WELL_NAME}' not found in dataset. Defaulting to first available well: {well_names[0]}")
    current_well = well_names[0]

fig = plt.figure(figsize=(18, 10))
fig.canvas.manager.set_window_title(f'Well Map & Similarity Explorer - Target: {current_well}')

gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.25)

ax_map = fig.add_subplot(gs[:, 0])      # Map view spanning both rows on the left
ax_z_md = fig.add_subplot(gs[0, 1])     # Z + TVT vs MD
ax_drxn = fig.add_subplot(gs[0, 2])     # Z + TVT vs Direction of Propagation
ax_proj = fig.add_subplot(gs[1, 1:])    # Neighbor TVT projection comparison on bottom right

def render_explorer():
    """Renders all analytical panels and updates spatial plots efficiently on demand."""
    ax_map.clear()
    ax_z_md.clear()
    ax_drxn.clear()
    ax_proj.clear()
    
    # Load target well on demand
    h1 = load_well_df(current_well)
    e1 = h1[h1['TVT_input'].isna()]
    if len(e1) == 0: 
        e1 = h1
        
    x1, y1_coord = e1['X'].iloc[0], e1['Y'].iloc[0]
    try:
        m1, c1 = np.polyfit(e1['MD'], e1['Z'], deg=1)
    except:
        m1 = 0.0
        
    y1_vals = e1['TVT'] + e1['Z']
    
    # Query top candidate neighbors using spatial KDTree
    distances, local_candidate_indices = start_tree.query([x1, y1_coord], k=min(25, len(well_names)))
    
    pos_results = []
    neg_results = []
    
    for idx, dist in zip(local_candidate_indices, distances):
        w2 = well_names[idx]
        if w2 == current_well:
            continue
        h2 = load_well_df(w2)
        e2 = h2[h2['TVT_input'].isna()]
        if len(e2) == 0: 
            e2 = h2
        
        sim_res = calculate_well_similarity(e2, e1)
        sim_res['well'] = w2
        
        if sim_res['parallelism'] > 0:
            pos_results.append(sim_res)
        else:
            neg_results.append(sim_res)
            
    top_results = (sorted(pos_results, key=lambda x: x['distance_ft']) + sorted(neg_results, key=lambda x: x['distance_ft']))[:TOP_K_NEIGHBORS]
    top_neighbor_names = {res['well'] for res in top_results}
    
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    ax_map.set_title(f'Map View (XY Plane)\nTarget: {current_well}', fontsize=11, fontweight='bold', color='#1f77b4')
    ax_map.set_xlabel('X Coordinate (ft/m)')
    ax_map.set_ylabel('Y Coordinate (ft/m)')
    ax_map.grid(True, linestyle='--', alpha=0.5)
    
    # Plot background well start points in neutral gray for maximum speed
    for idx, w in enumerate(well_names):
        if w == current_well or w in top_neighbor_names:
            continue
        sx, sy = start_points[idx]
        ax_map.scatter([sx], [sy], color='#d0d0d0', s=10, alpha=0.5)
            
    # Plot top neighbor wells with distinct highlighted colors
    neighbor_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, res in enumerate(top_results):
        w2 = res['well']
        h2 = load_well_df(w2)
        e2 = h2[h2['TVT_input'].isna()]
        if len(e2) == 0: e2 = h2
        col = neighbor_colors[i % len(neighbor_colors)]
        ax_map.plot(e2['X'], e2['Y'], color=col, linewidth=2.5, alpha=0.85, label=f'Top #{i+1}: {w2}')
        ax_map.scatter([e2['X'].iloc[0]], [e2['Y'].iloc[0]], color=col, s=70, edgecolors='k', zorder=5)

    # Plot selected target well prominently
    ax_map.plot(e1['X'], e1['Y'], color='dodgerblue', linewidth=3.5, zorder=6, label=f'Target: {current_well}')
    ax_map.scatter([x1], [y1_coord], color='red', s=140, marker='*', zorder=7, label='Target Start')
    
    # Dynamic Zoom: Focus map tightly around target well and its top neighbors with a 30% spatial buffer
    all_x = list(e1['X'])
    all_y = list(e1['Y'])
    for res in top_results:
        h2 = load_well_df(res['well'])
        e2 = h2[h2['TVT_input'].isna()]
        if len(e2) == 0: e2 = h2
        all_x.extend(e2['X'].tolist())
        all_y.extend(e2['Y'].tolist())
        
    x_margin = (max(all_x) - min(all_x)) * 0.3 if len(all_x) > 1 else 500
    y_margin = (max(all_y) - min(all_y)) * 0.3 if len(all_y) > 1 else 500
    if x_margin == 0: x_margin = 500
    if y_margin == 0: y_margin = 500
    
    ax_map.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax_map.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax_map.legend(loc='upper right', fontsize=8)
    
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    ax_z_md.set_title(f'Z + TVT vs MD (Top {TOP_K_NEIGHBORS} Neighbors)', fontsize=10, fontweight='bold')
    ax_z_md.plot(e1['MD'], y1_vals, label=f'Target: {current_well}', color='black', linewidth=2.5, zorder=5)
    ax_z_md.set_xlabel('MD')
    ax_z_md.set_ylabel('Z + TVT')
    ax_z_md.grid(True, linestyle='--', alpha=0.5)
    
    for res in top_results:
        w2 = res['well']
        h2 = load_well_df(w2)
        e2 = h2[h2['TVT_input'].isna()]
        if len(e2) == 0: e2 = h2
        y2_vals = e2['TVT'] + e2['Z']
        
        y2_interp = np.interp(e1['MD'], e2['MD'], y2_vals, left=np.nan, right=np.nan)
        rmse_val = safe_rmse(y1_vals.values, y2_interp)
        
        ax_z_md.plot(e2['MD'], y2_vals, label=f"{w2} (Dist:{res['distance_ft']:.0f}ft, RMSE:{rmse_val:.2f})", alpha=0.75, linewidth=1.8)
    ax_z_md.legend(loc='upper left', fontsize=7)
    
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    ax_drxn.set_title('Z + TVT vs Direction of Propagation', fontsize=10, fontweight='bold')
    theta = np.arctan2(e1["Y"].iloc[-1] - e1["Y"].iloc[0], e1["X"].iloc[-1] - e1["X"].iloc[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    X1_proj = (e1["X"] * cos_t + e1["Y"] * sin_t)
    
    ax_drxn.plot(X1_proj, y1_vals, label=f'Target: {current_well}', color='black', linewidth=2.5, zorder=5)
    ax_drxn.set_xlabel('Projected Direction (X*cos + Y*sin)')
    ax_drxn.set_ylabel('Z + TVT')
    ax_drxn.grid(True, linestyle='--', alpha=0.5)
    
    for res in top_results:
        w2 = res['well']
        h2 = load_well_df(w2)
        e2 = h2[h2['TVT_input'].isna()]
        if len(e2) == 0: e2 = h2
        X2_proj = (e2["X"] * cos_t + e2["Y"] * sin_t)
        y2_vals = e2['TVT'] + e2['Z']
        if res['parallelism'] > 0:
            ax_drxn.plot(X2_proj, y2_vals, label=f"{w2} (Sim:{res['parallelism']:.2f})", alpha=0.75, linewidth=1.8)
    ax_drxn.legend(loc='upper left', fontsize=7)
    
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    ax_proj.set_title('Projected Neighbor TVT Mapping onto Target Well', fontsize=10, fontweight='bold')
    y1_detrended = y1_vals - m1 * e1['MD']
    ax_proj.plot(e1['MD'], y1_detrended, label=f'Target Original: {current_well}', color='black', linewidth=2.5, zorder=5)
    ax_proj.set_xlabel('MD')
    ax_proj.set_ylabel('Detrended Z + TVT')
    ax_proj.grid(True, linestyle='--', alpha=0.5)
    
    for res in top_results:
        w2 = res['well']
        h2 = load_well_df(w2)
        e1_mapped = project_neighbor_tvt(e1, h2)
        y2_proj = e1_mapped['n_ztvt'] - m1 * e1['MD']
        y2_proj = y2_proj - y2_proj.iloc[0] + (y1_vals.iloc[0] - m1 * e1['MD'].iloc[0])
        
        # Calculate final RMSE for the projected mapping and display in legend
        proj_rmse = safe_rmse(y1_detrended.values, y2_proj.values)
        
        ax_proj.scatter(e1['MD'], y2_proj, c=e1_mapped['dist_n'], cmap='plasma', s=18, label=f'{w2} (Dist: {res["distance_ft"]:.0f}ft, RMSE: {proj_rmse:.2f})')
        
    ax_proj.legend(loc='upper left', fontsize=7)
    
    fig.canvas.draw_idle()

if __name__ == '__main__':
    print(f"Launching Well Map & Similarity Explorer for target well: {current_well}...")
    render_explorer()
    plt.show()