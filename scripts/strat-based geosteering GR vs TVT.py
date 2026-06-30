import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# --- 1. DATA PREPARATION ---
IS_KAGGLE = os.path.exists('/kaggle/input')
BASE_DIR = '/kaggle/input/competitions/rogii-wellbore-geology-prediction' if IS_KAGGLE else '../data'
train_data = os.path.join(BASE_DIR, "train")

def remove_rotation_noise(gr_series, cutoff):
    """
    Low pass filter.
    """
    # 1. Handle NaNs safely
    gr_filled = gr_series.interpolate(method='linear').bfill().ffill()
    
    # 2. Design the Butterworth Filter
    nyquist = 0.5 
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    
    # 3. Apply Zero-Phase Filtering
    gr_clean = filtfilt(b, a, gr_filled)
    return gr_clean

well = '1b1c5372'

# Load the data (Handle gracefully if user runs this outside of correct path)
try:
    hw = pd.read_csv(train_data + f"/{well}__horizontal_well.csv")
    tw = pd.read_csv(train_data + f"/{well}__typewell.csv")
except FileNotFoundError:
    print(f"ERROR: Could not find data at {train_data}. Please verify the path.")
    exit()

mask = hw['TVT_input'].isna()
evalz = hw[mask].copy()

HW_LOW_PASS_CUTOFF = 0.009
PEAK_PROMINENCE = 3.0
PEAK_DISTANCE = 50
EXTRA_DIVISIONS = [20, 70, 150]

fill = lambda df: df.interpolate(method='linear', limit_direction='both')
d = lambda df: fill(df).diff().rolling(11, center=True, min_periods=1).mean()

# Normalizing stuff
slope, intercept = np.polyfit(evalz['MD'], evalz['Z'], deg=1)
pred = evalz['Z'] - evalz['MD'] * slope - intercept
pred_norm = pred - pred.iloc[0] 
tvt_norm = -evalz['TVT'] + evalz['TVT'].iloc[0]
evalz['norm_TVT'] = tvt_norm
evalz['norm_Z'] = pred_norm
evalz['Z-bl'] = evalz['norm_TVT'] - evalz['norm_Z']
evalz['GR'] = remove_rotation_noise(evalz['GR'], cutoff=HW_LOW_PASS_CUTOFF)
evalz['dZ-bl'] = d(evalz['Z-bl'])


# --- 2. THE STANDALONE MATPLOTLIB APPLICATION (INTERP / Z PERSPECTIVE) ---
class StarSteerSimulator:
    def __init__(self, evalz, tw):
        self.evalz = evalz.copy()
        self.tw = tw.copy()
        
        # State tracking for chunks
        self.chunks = [] 
        self.current_md_start = self.evalz['MD'].iloc[0]
        self.current_tvt_0 = self.evalz['TVT'].iloc[0]
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create Main Figure
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.35) # Leave bottom space for UI widgets
        
        # --- Setup Initial Plot Lines (Z vs GR Perspective) ---
        # 1. Sensed HW
        self.line_sensed, = self.ax.plot(self.evalz['Z'], self.evalz['GR'], label='Sensed HW GR', color='black', linewidth=1.5)
        
        # 2. Geologist Truth (TW mapped via original HW TVT)
        truth_gr_full = np.interp(self.evalz['TVT'], self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
        self.line_geo, = self.ax.plot(self.evalz['Z'], truth_gr_full, label='Truth (TW via Geo TVT)', color='green', alpha=0.3, linewidth=2)
        
        # 3. Active Chunk placeholders (empty to start)
        self.line_pred, = self.ax.plot([], [], color='dodgerblue', linewidth=2, label='Predicted TW Mapping')
        self.scatter_anchor, = self.ax.plot([], [], 'go', markersize=8, label='Current Anchor')
        
        self.history_lines = []
        self.history_anchors = []
        
        # Format Graph
        self.ax.set_title("Interactive Geosteering | Initializing...", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("True Vertical Depth (Z)", fontsize=12)
        self.ax.set_ylabel("Gamma Ray (API)", fontsize=12)
        
        min_z, max_z = self.evalz['Z'].min(), self.evalz['Z'].max()
        buffer = abs(max_z - min_z) * 0.05
        if buffer == 0: buffer = 10
        self.ax.set_xlim(min_z - buffer, max_z + buffer)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # --- Create Widgets ---
        axcolor = 'lightgoldenrodyellow'
        max_md = self.evalz['MD'].max()
        initial_end = min(self.current_md_start + 200, max_md)
        
        # Sliders (MD, m, c)
        self.ax_md = plt.axes([0.15, 0.22, 0.65, 0.03], facecolor=axcolor)
        self.ax_m = plt.axes([0.15, 0.17, 0.65, 0.03], facecolor=axcolor)
        self.ax_c = plt.axes([0.15, 0.12, 0.65, 0.03], facecolor=axcolor)
        
        self.slider_md = Slider(self.ax_md, 'Active MD End', self.current_md_start + 1, max_md, valinit=initial_end, valstep=5)
        self.slider_m = Slider(self.ax_m, 'm (Apparent Dip)', -0.05, 0.05, valinit=0.0, valstep=0.001)
        self.slider_c = Slider(self.ax_c, 'c (Fault Offset)', -5.0, 5.0, valinit=0.0, valstep=0.1)
        
        # Buttons (Commit, Reset)
        self.ax_commit = plt.axes([0.15, 0.03, 0.15, 0.05])
        self.btn_commit = Button(self.ax_commit, 'Drop Anchor', color='lightgreen', hovercolor='palegreen')
        
        self.ax_reset = plt.axes([0.35, 0.03, 0.15, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Reset All', color='salmon', hovercolor='lightsalmon')
        
        # Checkboxes (Toggles)
        self.ax_toggles = plt.axes([0.55, 0.02, 0.35, 0.08], frameon=False)
        self.toggles = CheckButtons(self.ax_toggles, ['Show Geologist Mapping', 'Show Predicted Mapping'], [True, True])
        
        # Bind Events
        self.slider_md.on_changed(self.update_plot)
        self.slider_m.on_changed(self.update_plot)
        self.slider_c.on_changed(self.update_plot)
        self.btn_commit.on_clicked(self.commit_chunk)
        self.btn_reset.on_clicked(self.reset_all)
        self.toggles.on_clicked(self.toggle_visibility)
        
        # Initial draw
        self.update_plot(None)
        plt.show() # This launches the native Python desktop window!
        
    def toggle_visibility(self, label):
        if label == 'Show Geologist Mapping':
            self.line_geo.set_visible(not self.line_geo.get_visible())
        elif label == 'Show Predicted Mapping':
            vis = not self.line_pred.get_visible()
            self.line_pred.set_visible(vis)
            self.scatter_anchor.set_visible(vis)
            for line in self.history_lines:
                line.set_visible(vis)
            for anchor in self.history_anchors:
                anchor.set_visible(vis)
        self.fig.canvas.draw_idle()

    def _calculate_metrics(self, true_tvt, pred_tvt):
        """Helper to calculate Correlation and RMSE directly on TVT target error"""
        if len(true_tvt) == 0:
            return 0.0, 0.0
            
        t_tvt = np.array(true_tvt)
        p_tvt = np.array(pred_tvt)
        
        valid = ~np.isnan(t_tvt) & ~np.isnan(p_tvt)
        corr, rmse = 0.0, 0.0
        
        if np.sum(valid) > 5:
            # We now evaluate the TVT error directly!
            rmse = np.sqrt(np.mean((t_tvt[valid] - p_tvt[valid])**2))
            if np.std(t_tvt[valid]) > 1e-5 and np.std(p_tvt[valid]) > 1e-5:
                corr, _ = pearsonr(t_tvt[valid], p_tvt[valid])
                
        return corr, rmse

    def update_plot(self, val):
        active_md_end = self.slider_md.val
        active_m = self.slider_m.val
        active_c = self.slider_c.val
        
        # --- 1. Calculate History Metrics ---
        hist_corr, hist_rmse = 0.0, 0.0
        if self.chunks:
            hist_true_tvt_list = []
            hist_pred_tvt_list = []
            for ch in self.chunks:
                mask = (self.evalz['MD'] >= ch['md_start']) & (self.evalz['MD'] <= ch['md_end'])
                c_data = self.evalz[mask]
                norm_z_shift = c_data['norm_Z'] - c_data['norm_Z'].iloc[0]
                tvt_seg = ch['m'] * (c_data['MD'] - ch['md_start']) - norm_z_shift + ch['tvt_0'] + ch['c']
                
                # We now track the true TVT vs the predicted TVT
                hist_true_tvt_list.append(c_data['TVT'])
                hist_pred_tvt_list.append(tvt_seg)
                
            full_hist_true_tvt = np.concatenate(hist_true_tvt_list)
            full_hist_pred_tvt = np.concatenate(hist_pred_tvt_list)
            hist_corr, hist_rmse = self._calculate_metrics(full_hist_true_tvt, full_hist_pred_tvt)
        
        # --- 2. Evaluate Active Chunk ---
        valid_mask = (self.evalz['MD'] >= self.current_md_start) & (self.evalz['MD'] <= active_md_end)
        active_data = self.evalz[valid_mask]
        
        active_corr, active_rmse = 0.0, 0.0
        if len(active_data) > 0:
            active_norm_Z = active_data['norm_Z'] - active_data['norm_Z'].iloc[0]
            active_tvt = active_m * (active_data['MD'] - self.current_md_start) - active_norm_Z + self.current_tvt_0 + active_c
            
            # Map TW GR onto MD using the predicted TVT
            active_pred_gr = np.interp(active_tvt, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
            
            # Draw lines in Z space
            self.line_pred.set_data(active_data['Z'], active_pred_gr)
            self.scatter_anchor.set_data([active_data['Z'].iloc[0]], [active_pred_gr[0]])
            
            # Calculate Active Metrics (TVT True vs TVT Predicted)
            active_corr, active_rmse = self._calculate_metrics(active_data['TVT'], active_tvt)
            
        # Update Title
        title_str = "Interactive Geosteering (Z-Domain)"
        if self.chunks:
            title_str += f" | Hist TVT: R={hist_corr:.3f}, RMSE={hist_rmse:.2f} ft"
        if len(active_data) > 0:
            title_str += f" | Active TVT: R={active_corr:.3f}, RMSE={active_rmse:.2f} ft"
            
        self.ax.set_title(title_str, fontsize=14, fontweight='bold')
            
        self.fig.canvas.draw_idle()
        
    def commit_chunk(self, event):
        active_md_end = self.slider_md.val
        active_m = self.slider_m.val
        active_c = self.slider_c.val
        
        valid_mask = (self.evalz['MD'] >= self.current_md_start) & (self.evalz['MD'] <= active_md_end)
        active_data = self.evalz[valid_mask]
        
        active_norm_Z = active_data['norm_Z'] - active_data['norm_Z'].iloc[0]
        active_tvt = active_m * (active_data['MD'] - self.current_md_start) - active_norm_Z + self.current_tvt_0 + active_c
        active_pred_gr = np.interp(active_tvt, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
        
        # Maintain toggle state for newly committed chunk
        vis = self.line_pred.get_visible()
        
        # Create a static line on the plot locking in the chunk (in Z space)
        line, = self.ax.plot(active_data['Z'], active_pred_gr, color='navy', alpha=0.7, linewidth=1.5, visible=vis)
        anchor, = self.ax.plot([active_data['Z'].iloc[-1]], [active_pred_gr[-1]], 'ro', markersize=6, visible=vis)
        
        self.history_lines.append(line)
        self.history_anchors.append(anchor)
        
        # Save History
        self.chunks.append({
            'md_start': self.current_md_start,
            'md_end': active_md_end,
            'm': active_m,
            'c': active_c,
            'tvt_0': self.current_tvt_0
        })
        
        # Handoff to Next Chunk
        self.current_md_start = active_md_end
        self.current_tvt_0 = active_tvt.iloc[-1]
        
        # Reset Sliders
        max_md = self.evalz['MD'].max()
        if self.current_md_start < max_md:
            self.slider_md.valmin = self.current_md_start + 1
            self.slider_md.ax.set_xlim(self.slider_md.valmin, self.slider_md.valmax)
            self.slider_md.set_val(min(self.current_md_start + 200, max_md))
        self.slider_m.set_val(0.0)
        self.slider_c.set_val(0.0)
        
    def reset_all(self, event):
        self.chunks = []
        self.current_md_start = self.evalz['MD'].iloc[0]
        self.current_tvt_0 = self.evalz['TVT'].iloc[0]
        
        # Remove visual history
        for line in self.history_lines:
            line.remove()
        for anchor in self.history_anchors:
            anchor.remove()
        self.history_lines.clear()
        self.history_anchors.clear()
        
        # Reset Sliders
        self.slider_md.valmin = self.current_md_start + 1
        self.slider_md.ax.set_xlim(self.slider_md.valmin, self.slider_md.valmax)
        self.slider_md.set_val(min(self.current_md_start + 200, self.evalz['MD'].max()))
        self.slider_m.set_val(0.0)
        self.slider_c.set_val(0.0)

# --- 3. EXECUTE APP ---
if __name__ == '__main__':
    steerer = StarSteerSimulator(evalz, tw)