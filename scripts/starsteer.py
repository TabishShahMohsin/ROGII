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
    """Low pass filter."""
    gr_filled = gr_series.interpolate(method='linear').bfill().ffill()
    nyquist = 0.5 
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, gr_filled)

well = '7bb17b96'

try:
    hw = pd.read_csv(train_data + f"/{well}__horizontal_well.csv")
    tw = pd.read_csv(train_data + f"/{well}__typewell.csv")
except FileNotFoundError:
    print(f"ERROR: Could not find data at {train_data}. Please verify the path.")
    exit()

HW_LOW_PASS_CUTOFF = 0.009

mask = hw['TVT_input'].isna()
norm_hw = hw[~mask].iloc[-1000:].copy()
hw_gr_calib = remove_rotation_noise(norm_hw['GR'], HW_LOW_PASS_CUTOFF)
hw_tvt_calib = norm_hw['TVT_input']
tw_gr_calib = np.interp(hw_tvt_calib, tw['TVT'], tw['GR'])

# Calculate the mean and standard deviation for the overlapping section
hw_mean, hw_std = np.mean(hw_gr_calib), np.std(hw_gr_calib)
tw_mean, tw_std = np.mean(tw_gr_calib), np.std(tw_gr_calib)

evalz = hw[mask].copy()
evalz['GR_raw'] = (evalz['GR'].copy() - hw_mean) / hw_std * tw_std + tw_mean



fill = lambda df: df.interpolate(method='linear', limit_direction='both')
d = lambda df: fill(df).diff().rolling(11, center=True, min_periods=1).mean()

# Normalizing Z and computing pure TVD
slope, intercept = np.polyfit(evalz['MD'], evalz['Z'], deg=1)
pred = evalz['Z'] - evalz['MD'] * slope - intercept
evalz['norm_Z'] = pred - pred.iloc[0] 

# Keep a raw copy of GR to allow dynamic filtering via slider
evalz['GR_raw'] = evalz['GR'].copy()
evalz['GR'] = remove_rotation_noise(evalz['GR_raw'], cutoff=HW_LOW_PASS_CUTOFF)

# STARSTEER STANDARD: TVD increases downwards, so TVD = -Z
evalz['TVD'] = -evalz['Z']


# --- 2. THE STANDALONE MATPLOTLIB APPLICATION (STARSTEER LAYOUT) ---
class StarSteerSimulator:
    def __init__(self, evalz, tw):
        self.evalz = evalz.copy()
        self.tw = tw.copy()
        
        # State tracking for chunks
        self.chunks = [] 
        self.current_md_start = self.evalz['MD'].iloc[0]
        self.current_tvt_0 = self.evalz['TVT'].iloc[0]
        self.current_cutoff = HW_LOW_PASS_CUTOFF
        
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_ax = None
        self._syncing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create Main Figure
        self.fig = plt.figure(figsize=(16, 9))
        
        # Adjust layout to make room for sliders in the bottom left
        plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.92) 
        
        # --- Create 3-Panel StarSteer GridSpec ---
        # 3 rows. The 3rd row is empty on the left (for sliders).
        gs = self.fig.add_gridspec(3, 2, width_ratios=[4, 1], height_ratios=[1, 3, 1.8], wspace=0.05, hspace=0.05)
        
        # Initialize the interconnected axes
        self.ax_top = self.fig.add_subplot(gs[0, 0])
        self.ax_main = self.fig.add_subplot(gs[1, 0], sharex=self.ax_top) # Shared X (MD)
        
        # THIS SPANS ROWS 0 & 1: Aligns perfectly with top of top graph and bottom of main graph!
        self.ax_right = self.fig.add_subplot(gs[0:2, 1]) 
        
        # Hide inner tick labels for a clean, unified grid
        plt.setp(self.ax_top.get_xticklabels(), visible=False)
        plt.setp(self.ax_right.get_yticklabels(), visible=False)
        
        # Invert the Y axes so True Vertical Depth (TVD) increases downwards
        self.ax_main.invert_yaxis()
        self.ax_right.invert_yaxis()
        
        # We pre-calculate the truth mapping once
        truth_gr_full = np.interp(self.evalz['TVT'], self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
        
        # --- TOP PANEL (MD vs GR) ---
        self.line_sensed_top, = self.ax_top.plot(self.evalz['MD'], self.evalz['GR'], label='Sensed HW (Raw drill bit GR)', color='black', linewidth=1.5)
        self.line_geo_top, = self.ax_top.plot(self.evalz['MD'], truth_gr_full, label='Truth TW (Ref GR mapped via Geo TVT)', color='green', alpha=0.3, linewidth=2)
        self.line_pred_top, = self.ax_top.plot([], [], color='dodgerblue', linewidth=2, label='Predicted TW (Ref GR mapped via Active m, c)')
        self.scatter_anchor_top, = self.ax_top.plot([], [], 'go', markersize=8, label='Current Anchor')
        
        # --- MAIN PANEL (MD vs TVD Cross-Section) ---
        self.line_traj_main, = self.ax_main.plot(self.evalz['MD'], self.evalz['TVD'], color='darkgrey', linewidth=1.0, zorder=1, label='Wellbore Trajectory (TVD = -Z)')
        self.line_pred_main, = self.ax_main.plot([], [], color='dodgerblue', linewidth=4, alpha=0.7, label='Active Predict Window', zorder=3)
        
        # --- RIGHT PANEL (GR vs TVD Stratigraphy) ---
        self.line_sensed_right, = self.ax_right.plot(self.evalz['GR'], self.evalz['TVD'], label='Sensed HW (Drill GR vs TVD)', color='black', linewidth=1.5)
        self.line_geo_right, = self.ax_right.plot(truth_gr_full, self.evalz['TVD'], label='Truth TW (Geo TVT mapped to TVD)', color='green', alpha=0.3, linewidth=2)
        self.line_pred_right, = self.ax_right.plot([], [], color='dodgerblue', linewidth=2, label='Predicted TW (Active TVT mapped to TVD)')
        
        self.history_lines_top = []
        self.history_lines_main = []
        self.history_lines_right = []
        self.history_anchors_top = []
        
        # Format Graphs
        self.fig.suptitle("Interactive Geosteering | Initializing...", fontsize=13, fontweight='bold')
        self.ax_top.set_ylabel("GR (API)", fontsize=10)
        self.ax_main.set_xlabel("Measured Depth (MD)", fontsize=12)
        self.ax_main.set_ylabel("True Vertical Depth (TVD)", fontsize=12)
        self.ax_right.set_xlabel("GR (API)", fontsize=12)
        
        self.ax_top.grid(True, alpha=0.3)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_right.grid(True, alpha=0.3)
        
        # Add explicit legends to all panels
        self.ax_top.legend(loc='upper right', fontsize=8)
        self.ax_main.legend(loc='upper right', fontsize=8)
        self.ax_right.legend(loc='upper right', fontsize=8)
        
        # Set Initial X Limits
        min_md, max_md = self.evalz['MD'].min(), self.evalz['MD'].max()
        self.ax_top.set_xlim(min_md - 50, max_md + 50)
        self.ax_right.set_xlim(50, 200)
        
        # --- Create Widgets (Safely positioned in the bottom left empty space) ---
        axcolor = 'lightgoldenrodyellow'
        initial_end = min(self.current_md_start + 200, max_md)
        
        # Sliders
        self.ax_md = plt.axes([0.10, 0.26, 0.40, 0.03], facecolor=axcolor)
        self.ax_m = plt.axes([0.10, 0.21, 0.40, 0.03], facecolor=axcolor)
        self.ax_c = plt.axes([0.10, 0.16, 0.40, 0.03], facecolor=axcolor)
        self.ax_cutoff = plt.axes([0.10, 0.11, 0.40, 0.03], facecolor=axcolor)
        
        self.slider_md = Slider(self.ax_md, 'Active MD End', self.current_md_start + 1, max_md, valinit=initial_end, valstep=5)
        self.slider_m = Slider(self.ax_m, 'm (Apparent Dip)', -0.05, 0.05, valinit=0.0, valstep=0.001)
        self.slider_c = Slider(self.ax_c, 'c (Fault Offset)', -5.0, 5.0, valinit=0.0, valstep=0.1)
        self.slider_cutoff = Slider(self.ax_cutoff, 'GR LP Cutoff', 0.001, 0.05, valinit=HW_LOW_PASS_CUTOFF, valstep=0.001)
        
        # Buttons
        self.ax_commit = plt.axes([0.10, 0.04, 0.15, 0.05])
        self.btn_commit = Button(self.ax_commit, 'Drop Anchor', color='lightgreen', hovercolor='palegreen')
        
        self.ax_reset = plt.axes([0.28, 0.04, 0.15, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Reset All', color='salmon', hovercolor='lightsalmon')
        
        # Toggles
        self.ax_toggles = plt.axes([0.55, 0.18, 0.22, 0.10], frameon=False)
        self.toggles = CheckButtons(self.ax_toggles, ['Show Geologist Mapping', 'Show Predicted Mapping'], [True, True])
        
        self.ax_lp_toggle = plt.axes([0.55, 0.10, 0.15, 0.05], frameon=False)
        self.lp_toggle = CheckButtons(self.ax_lp_toggle, ['Enable LP Filter'], [True])
        
        self.ax_legend_toggle = plt.axes([0.55, 0.04, 0.15, 0.05], frameon=False)
        self.legend_toggle = CheckButtons(self.ax_legend_toggle, ['Show Legends'], [True])
        
        # --- Mathematical Alignment Engine ---
        self.ax_main.callbacks.connect('ylim_changed', self.sync_y_from_main)
        self.ax_right.callbacks.connect('ylim_changed', self.sync_y_from_right)
        
        # Connect Mouse/Touchpad Gestures
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Bind UI Events
        self.slider_md.on_changed(self.update_plot)
        self.slider_m.on_changed(self.update_plot)
        self.slider_c.on_changed(self.update_plot)
        self.slider_cutoff.on_changed(self.update_plot)
        self.btn_commit.on_clicked(self.commit_chunk)
        self.btn_reset.on_clicked(self.reset_all)
        self.toggles.on_clicked(self.toggle_visibility)
        self.lp_toggle.on_clicked(self.toggle_lp_filter)
        self.legend_toggle.on_clicked(self.toggle_legends)
        
        # --- INIT LIMITS AND CALC BBOXES ---
        # Draw once to compute physical bboxes for the sync engine
        self.fig.canvas.draw()
        
        min_tvd, max_tvd = self.evalz['TVD'].min(), self.evalz['TVD'].max()
        buffer_tvd = abs(max_tvd - min_tvd) * 0.05
        if buffer_tvd == 0: buffer_tvd = 10
        # Set main plot limits. This triggers sync_y_from_main, perfectly aligning ax_right!
        self.ax_main.set_ylim(max_tvd + buffer_tvd, min_tvd - buffer_tvd) 

        # Save initial limits to properly reset the zoom later
        self.initial_xlim = self.ax_top.get_xlim()
        self.initial_ylim = self.ax_main.get_ylim()
        
        self.update_plot(None)
        plt.show() 

    # --- THE ALIGNMENT ENGINE ---
    # Keeps horizontal grids perfectly matched across axes of different physical heights
    def sync_y_from_main(self, ax=None):
        if self._syncing: return
        self._syncing = True
        bbox_main = self.ax_main.get_position()
        bbox_right = self.ax_right.get_position()
        y_bottom, y_top = self.ax_main.get_ylim() 
        dy_dfig = (y_top - y_bottom) / (bbox_main.y1 - bbox_main.y0)
        new_right_bottom = y_bottom + dy_dfig * (bbox_right.y0 - bbox_main.y0)
        new_right_top = y_top + dy_dfig * (bbox_right.y1 - bbox_main.y1)
        self.ax_right.set_ylim(new_right_bottom, new_right_top)
        self._syncing = False

    def sync_y_from_right(self, ax=None):
        if self._syncing: return
        self._syncing = True
        bbox_main = self.ax_main.get_position()
        bbox_right = self.ax_right.get_position()
        y_bottom, y_top = self.ax_right.get_ylim() 
        dy_dfig = (y_top - y_bottom) / (bbox_right.y1 - bbox_right.y0)
        new_main_bottom = y_bottom + dy_dfig * (bbox_main.y0 - bbox_right.y0)
        new_main_top = y_top + dy_dfig * (bbox_main.y1 - bbox_right.y1)
        self.ax_main.set_ylim(new_main_bottom, new_main_top)
        self._syncing = False

    # --- TOUCHPAD GESTURES & ZOOM/PAN ---
    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None: return
        
        # Smooth zoom for high-res touchpads using step velocity
        scale_factor = 1.15 ** (-event.step)
        
        # Zoom X
        x_min, x_max = ax.get_xlim()
        x_range = (x_max - x_min) * scale_factor
        x_ratio = (event.xdata - x_min) / (x_max - x_min)
        ax.set_xlim(event.xdata - x_range * x_ratio, event.xdata + x_range * (1 - x_ratio))
        
        # Zoom Y
        y_min, y_max = ax.get_ylim()
        y_range = (y_max - y_min) * scale_factor
        y_ratio = (event.ydata - y_min) / (y_max - y_min)
        ax.set_ylim(event.ydata - y_range * y_ratio, event.ydata + y_range * (1 - y_ratio))
        
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.button in [2, 3] and event.inaxes in [self.ax_top, self.ax_main, self.ax_right]: 
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_ax = event.inaxes
            self.xlim_start = event.inaxes.get_xlim()
            self.ylim_start = event.inaxes.get_ylim()

    def on_release(self, event):
        if event.button in [2, 3]:
            self.pan_start_x = None

    def on_motion(self, event):
        if self.pan_start_x is None or event.inaxes != self.pan_ax:
            return
        
        ax = self.pan_ax
        dx_pixels = event.x - self.pan_start_x
        dy_pixels = event.y - self.pan_start_y
        
        bbox = ax.get_window_extent()
        dx_data = dx_pixels * (self.xlim_start[1] - self.xlim_start[0]) / bbox.width
        dy_data = dy_pixels * (self.ylim_start[1] - self.ylim_start[0]) / bbox.height
        
        ax.set_xlim(self.xlim_start[0] - dx_data, self.xlim_start[1] - dx_data)
        ax.set_ylim(self.ylim_start[0] - dy_data, self.ylim_start[1] - dy_data)
        self.fig.canvas.draw_idle()
        
    def toggle_visibility(self, label):
        if label == 'Show Geologist Mapping':
            self.line_geo_top.set_visible(not self.line_geo_top.get_visible())
            self.line_geo_right.set_visible(not self.line_geo_right.get_visible())
        elif label == 'Show Predicted Mapping':
            vis = not self.line_pred_top.get_visible()
            self.line_pred_top.set_visible(vis)
            self.line_pred_main.set_visible(vis)
            self.line_pred_right.set_visible(vis)
            self.scatter_anchor_top.set_visible(vis)
            for line in self.history_lines_top: line.set_visible(vis)
            for line in self.history_lines_main: line.set_visible(vis)
            for line in self.history_lines_right: line.set_visible(vis)
            for anchor in self.history_anchors_top: anchor.set_visible(vis)
        self.fig.canvas.draw_idle()

    def toggle_legends(self, label):
        vis = self.legend_toggle.get_status()[0]
        if self.ax_top.get_legend(): self.ax_top.get_legend().set_visible(vis)
        if self.ax_main.get_legend(): self.ax_main.get_legend().set_visible(vis)
        if self.ax_right.get_legend(): self.ax_right.get_legend().set_visible(vis)
        self.fig.canvas.draw_idle()

    def toggle_lp_filter(self, label):
        self._apply_gr_filter()
        self.update_plot(None)
        
    def _apply_gr_filter(self):
        if self.lp_toggle.get_status()[0]:
            self.evalz['GR'] = remove_rotation_noise(self.evalz['GR_raw'], self.current_cutoff)
        else:
            self.evalz['GR'] = self.evalz['GR_raw']
        self.line_sensed_top.set_ydata(self.evalz['GR'])
        self.line_sensed_right.set_xdata(self.evalz['GR'])

    def _calculate_metrics(self, true_arr, pred_arr):
        """Generic helper to calculate Correlation and RMSE"""
        if len(true_arr) == 0:
            return 0.0, 0.0
            
        t_arr = np.array(true_arr)
        p_arr = np.array(pred_arr)
        
        valid = ~np.isnan(t_arr) & ~np.isnan(p_arr)
        corr, rmse = 0.0, 0.0
        
        if np.sum(valid) > 5:
            rmse = np.sqrt(np.mean((t_arr[valid] - p_arr[valid])**2))
            if np.std(t_arr[valid]) > 1e-5 and np.std(p_arr[valid]) > 1e-5:
                corr, _ = pearsonr(t_arr[valid], p_arr[valid])
                
        return corr, rmse

    def update_plot(self, val):
        active_md_end = self.slider_md.val
        active_m = self.slider_m.val
        active_c = self.slider_c.val
        active_cutoff = self.slider_cutoff.val
        
        # Check if the cutoff slider was moved, recalculate GR smoothing if it was
        if abs(active_cutoff - self.current_cutoff) > 1e-6:
            self.current_cutoff = active_cutoff
            self._apply_gr_filter()
        
        # --- 1. Calculate History Metrics ---
        hist_tvt_corr, hist_tvt_rmse = 0.0, 0.0
        hist_gr_corr, hist_gr_rmse = 0.0, 0.0
        
        if self.chunks:
            hist_true_tvt_list, hist_pred_tvt_list = [], []
            hist_sensed_gr_list, hist_pred_gr_list = [], []
            
            for ch in self.chunks:
                mask = (self.evalz['MD'] >= ch['md_start']) & (self.evalz['MD'] <= ch['md_end'])
                c_data = self.evalz[mask]
                norm_z_shift = c_data['norm_Z'] - c_data['norm_Z'].iloc[0]
                tvt_seg = ch['m'] * (c_data['MD'] - ch['md_start']) - norm_z_shift + ch['tvt_0'] + ch['c']
                
                # Interpolate TW onto MD
                pred_gr_seg = np.interp(tvt_seg, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
                
                hist_true_tvt_list.append(c_data['TVT'])
                hist_pred_tvt_list.append(tvt_seg)
                hist_sensed_gr_list.append(c_data['GR'])
                hist_pred_gr_list.append(pred_gr_seg)
                
            full_hist_true_tvt = np.concatenate(hist_true_tvt_list)
            full_hist_pred_tvt = np.concatenate(hist_pred_tvt_list)
            full_hist_sensed_gr = np.concatenate(hist_sensed_gr_list)
            full_hist_pred_gr = np.concatenate(hist_pred_gr_list)
            
            hist_tvt_corr, hist_tvt_rmse = self._calculate_metrics(full_hist_true_tvt, full_hist_pred_tvt)
            hist_gr_corr, hist_gr_rmse = self._calculate_metrics(full_hist_sensed_gr, full_hist_pred_gr)
        
        # --- 2. Evaluate Active Chunk ---
        valid_mask = (self.evalz['MD'] >= self.current_md_start) & (self.evalz['MD'] <= active_md_end)
        active_data = self.evalz[valid_mask]
        
        active_tvt_corr, active_tvt_rmse = 0.0, 0.0
        active_gr_corr, active_gr_rmse = 0.0, 0.0
        
        if len(active_data) > 0:
            active_norm_Z = active_data['norm_Z'] - active_data['norm_Z'].iloc[0]
            active_tvt = active_m * (active_data['MD'] - self.current_md_start) - active_norm_Z + self.current_tvt_0 + active_c
            
            # Map TW GR onto MD using the predicted TVT 
            active_pred_gr = np.interp(active_tvt, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
            
            # Update Top Plot (MD vs GR)
            self.line_pred_top.set_data(active_data['MD'], active_pred_gr)
            self.scatter_anchor_top.set_data([active_data['MD'].iloc[0]], [active_pred_gr[0]])
            
            # Update Main Plot (MD vs TVD trajectory highlight)
            self.line_pred_main.set_data(active_data['MD'], active_data['TVD'])
            
            # Update Right Plot (GR vs TVD)
            self.line_pred_right.set_data(active_pred_gr, active_data['TVD'])
            
            # Calculate Active Metrics
            active_tvt_corr, active_tvt_rmse = self._calculate_metrics(active_data['TVT'], active_tvt)
            active_gr_corr, active_gr_rmse = self._calculate_metrics(active_data['GR'], active_pred_gr)
            
        # Update Title with Dual Metrics
        title_str = "Interactive StarSteer Mode (Use Trackpad or Right-Click to Pan/Zoom)\n"
        if self.chunks:
            title_str += f"Hist    |   TVT: R={hist_tvt_corr:.3f}, RMSE={hist_tvt_rmse:.2f} ft   ||   GR: R={hist_gr_corr:.3f}, RMSE={hist_gr_rmse:.2f} API\n"
        else:
            title_str += "Hist    |   No chunks committed yet\n"
        if len(active_data) > 0:
            title_str += f"Active |   TVT: R={active_tvt_corr:.3f}, RMSE={active_tvt_rmse:.2f} ft   ||   GR: R={active_gr_corr:.3f}, RMSE={active_gr_rmse:.2f} API"
            
        self.fig.suptitle(title_str, fontsize=12, fontweight='bold')
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
        
        vis = self.line_pred_top.get_visible()
        
        # Create static lines on all 3 plots locking in the chunk (Matched colors)
        l_top, = self.ax_top.plot(active_data['MD'], active_pred_gr, color='navy', alpha=0.7, linewidth=1.5, visible=vis)
        l_main, = self.ax_main.plot(active_data['MD'], active_data['TVD'], color='navy', alpha=0.7, linewidth=4, visible=vis)
        l_right, = self.ax_right.plot(active_pred_gr, active_data['TVD'], color='navy', alpha=0.7, linewidth=1.5, visible=vis)
        anchor, = self.ax_top.plot([active_data['MD'].iloc[-1]], [active_pred_gr[-1]], 'ro', markersize=6, visible=vis)
        
        self.history_lines_top.append(l_top)
        self.history_lines_main.append(l_main)
        self.history_lines_right.append(l_right)
        self.history_anchors_top.append(anchor)
        
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
        for lst in [self.history_lines_top, self.history_lines_main, self.history_lines_right, self.history_anchors_top]:
            for item in lst:
                item.remove()
            lst.clear()
        
        # Reset Sliders
        self.slider_md.valmin = self.current_md_start + 1
        self.slider_md.ax.set_xlim(self.slider_md.valmin, self.slider_md.valmax)
        self.slider_md.set_val(min(self.current_md_start + 200, self.evalz['MD'].max()))
        self.slider_m.set_val(0.0)
        self.slider_c.set_val(0.0)
        self.slider_cutoff.set_val(HW_LOW_PASS_CUTOFF)
        
        # Restore original Plot Zoom/Pan
        self.ax_top.set_xlim(self.initial_xlim)
        self.ax_main.set_ylim(self.initial_ylim)

# --- 3. EXECUTE APP ---
if __name__ == '__main__':
    steerer = StarSteerSimulator(evalz, tw)