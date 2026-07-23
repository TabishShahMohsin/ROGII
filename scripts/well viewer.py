import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
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

well = '276b012a'

try:
    hw = pd.read_csv(train_data + f"/{well}__horizontal_well.csv")
    tw = pd.read_csv(train_data + f"/{well}__typewell.csv")
except FileNotFoundError:
    print(f"ERROR: Could not find data at {train_data}. Please verify the path.")
    exit()

HW_LOW_PASS_CUTOFF = 0.009

# CALCULATE THL (True Horizontal Length) based on X/Y Coordinates or MD fallback on entire well
if 'X' in hw.columns and 'Y' in hw.columns:
    hw['THL'] = np.sqrt((hw['X'] - hw['X'].iloc[0])**2 + (hw['Y'] - hw['Y'].iloc[0])**2)
else:
    # Fallback approximation for horizontal length if X/Y aren't explicitly provided
    dz = hw['Z'].diff().fillna(0)
    dmd = hw['MD'].diff().fillna(0)
    dthl = np.sqrt(np.maximum(dmd**2 - dz**2, 0))
    hw['THL'] = dthl.cumsum()

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

evalz['GR'] = remove_rotation_noise(evalz['GR_raw'], cutoff=HW_LOW_PASS_CUTOFF)

# STANDARD: TVD increases downwards, so TVD = -Z
evalz['TVD'] = -evalz['Z']

# --- 2. DUAL-WINDOW GEOSTEERING SIMULATOR ---
class GeosteeringSimulator:
    def __init__(self, evalz, tw):
        self.evalz = evalz.copy()
        self.tw = tw.copy()
        
        # State tracking for chunks and parameters
        self.chunks = [] 
        self.current_md_start = self.evalz['MD'].iloc[0]
        self.current_tvt_0 = self.evalz['TVT'].iloc[0] if 'TVT' in self.evalz.columns else 0.0
        self.current_cutoff = HW_LOW_PASS_CUTOFF
        
        # State tracking for Dynamic Axes (MD vs THL, TVD vs TVT)
        self.top_axis_mode = 'MD'
        self.right_axis_mode = 'TVD'
        
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_ax = None
        self._syncing = False
        
        # Pre-calculate the geological truth mapping for TVD correlation mode
        if 'TVT' in self.evalz.columns:
            self.truth_gr_full = np.interp(self.evalz['TVT'], self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
        else:
            self.truth_gr_full = np.full(len(self.evalz), np.nan)
            
        self.setup_ui()
        
    def setup_ui(self):
        # =========================================================================
        # WINDOW 1: GRAPH & METRICS DISPLAY
        # =========================================================================
        self.fig_plot = plt.figure("StarSteer Visualizer", figsize=(14, 8))
        self.fig_plot.subplots_adjust(bottom=0.08, left=0.07, right=0.95, top=0.88, wspace=0.05, hspace=0.05)
        
        # Clean 2x2 GridSpec filling the full window (Top: MD vs GR, Bottom: MD vs TVD, Right: Stratigraphy)
        gs = self.fig_plot.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.05)
        
        self.ax_top = self.fig_plot.add_subplot(gs[0, 0])
        self.ax_main = self.fig_plot.add_subplot(gs[1, 0], sharex=self.ax_top)
        self.ax_right = self.fig_plot.add_subplot(gs[0:2, 1]) 
        
        # Hide inner tick labels
        plt.setp(self.ax_top.get_xticklabels(), visible=False)
        plt.setp(self.ax_right.get_yticklabels(), visible=False)
        
        # Invert Y axes so TVD/TVT increases downwards
        self.ax_main.invert_yaxis()
        self.ax_right.invert_yaxis()
        
        # Top Panel
        self.line_sensed_top, = self.ax_top.plot(self.evalz['MD'], self.evalz['GR'], label='Sensed HW', color='black', linewidth=1.5)
        self.line_geo_top, = self.ax_top.plot(self.evalz['MD'], self.truth_gr_full, label='Truth TW', color='green', alpha=0.3, linewidth=2)
        self.line_pred_top, = self.ax_top.plot([], [], color='dodgerblue', linewidth=2, label='Predicted TW')
        self.scatter_anchor_top, = self.ax_top.plot([], [], 'go', markersize=8, label='Current Anchor')
        
        # Main Panel
        self.line_traj_main, = self.ax_main.plot(self.evalz['MD'], self.evalz['TVD'], color='darkgrey', linewidth=1.0, zorder=1, label='Trajectory')
        self.line_pred_main, = self.ax_main.plot([], [], color='dodgerblue', linewidth=4, alpha=0.7, label='Active Window', zorder=3)
        
        # Right Panel
        self.line_sensed_right, = self.ax_right.plot(self.evalz['GR'], self.evalz['TVD'], label='Sensed HW', color='black', linewidth=1.5)
        self.line_geo_right, = self.ax_right.plot(self.truth_gr_full, self.evalz['TVD'], label='Truth TW', color='green', alpha=0.3, linewidth=2)
        self.line_pred_right, = self.ax_right.plot([], [], color='dodgerblue', linewidth=2, label='Predicted TW')
        
        self.history_lines_top = []
        self.history_lines_main = []
        self.history_lines_right = []
        self.history_anchors_top = []
        
        # Formatting
        self.fig_plot.suptitle("Interactive Geosteering | Initializing...", fontsize=12, fontweight='bold')
        self.ax_top.set_ylabel("GR (API)", fontsize=10)
        self.ax_main.set_xlabel("Measured Depth (MD)", fontsize=11)
        self.ax_main.set_ylabel("True Vertical Depth (TVD)", fontsize=11)
        self.ax_right.set_xlabel("GR (API)", fontsize=11)
        self.ax_right.set_title("Correlation Panel (TVD)", fontsize=10)
        
        self.ax_top.grid(True, alpha=0.3)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_right.grid(True, alpha=0.3)
        
        self.ax_top.legend(loc='upper right', fontsize=8)
        self.ax_main.legend(loc='upper right', fontsize=8)
        self.ax_right.legend(loc='upper right', fontsize=8)
        
        min_md, max_md = self.evalz['MD'].min(), self.evalz['MD'].max()
        self.ax_top.set_xlim(min_md - 50, max_md + 50)
        
        min_gr = min(self.evalz['GR'].min(), self.tw['GR'].min())
        max_gr = max(self.evalz['GR'].max(), self.tw['GR'].max())
        self.ax_right.set_xlim(min_gr - 10, max_gr + 10)

        # =========================================================================
        # WINDOW 2: CONTROL PANEL (SLIDERS, BUTTONS, CHECKBOXES)
        # =========================================================================
        self.fig_ctrl = plt.figure("Geosteering Controls", figsize=(5.5, 9))
        self.fig_ctrl.suptitle("Geosteering Controls", fontsize=13, fontweight='bold', y=0.96)
        
        axcolor = '#e9ecef'
        initial_end = min(self.current_md_start + 200, max_md)
        
        # --- Section 1: Parameter Sliders ---
        self.ax_md     = self.fig_ctrl.add_axes([0.30, 0.85, 0.62, 0.035], facecolor=axcolor)
        self.ax_m      = self.fig_ctrl.add_axes([0.30, 0.78, 0.62, 0.035], facecolor=axcolor)
        self.ax_c      = self.fig_ctrl.add_axes([0.30, 0.71, 0.62, 0.035], facecolor=axcolor)
        self.ax_cutoff = self.fig_ctrl.add_axes([0.30, 0.64, 0.62, 0.035], facecolor=axcolor)
        
        self.slider_md     = Slider(self.ax_md, 'Active End', self.current_md_start + 1, max_md, valinit=initial_end, valstep=5)
        self.slider_m      = Slider(self.ax_m, 'm (App. Dip)', -0.15, 0.15, valinit=0.0, valstep=0.001)
        self.slider_c      = Slider(self.ax_c, 'c (Offset)', -5.0, 5.0, valinit=0.0, valstep=0.1)
        self.slider_cutoff = Slider(self.ax_cutoff, 'LP Cutoff', 0.001, 0.05, valinit=HW_LOW_PASS_CUTOFF, valstep=0.001)
        
        # --- Section 2: Action Buttons ---
        self.ax_commit = self.fig_ctrl.add_axes([0.10, 0.54, 0.25, 0.06])
        self.btn_commit = Button(self.ax_commit, 'Drop Anchor', color='lightgreen', hovercolor='palegreen')
        
        self.ax_undo = self.fig_ctrl.add_axes([0.38, 0.54, 0.25, 0.06])
        self.btn_undo = Button(self.ax_undo, 'Undo', color='khaki', hovercolor='palegoldenrod')
        
        self.ax_reset = self.fig_ctrl.add_axes([0.66, 0.54, 0.24, 0.06])
        self.btn_reset = Button(self.ax_reset, 'Reset All', color='salmon', hovercolor='lightsalmon')
        
        # --- Section 3: Axis Toggles (MD/THL and TVD/TVT) ---
        self.ax_radio_x = self.fig_ctrl.add_axes([0.10, 0.43, 0.38, 0.08], facecolor=axcolor)
        self.ax_radio_x.set_title("X-Axis (Top & Main)", fontsize=9, pad=4, loc='left', fontweight='bold')
        self.radio_x = RadioButtons(self.ax_radio_x, ('MD', 'THL'))
        
        self.ax_radio_y = self.fig_ctrl.add_axes([0.52, 0.43, 0.38, 0.08], facecolor=axcolor)
        self.ax_radio_y.set_title("Y-Axis (Main & Right)", fontsize=9, pad=4, loc='left', fontweight='bold')
        self.radio_y = RadioButtons(self.ax_radio_y, ('TVD', 'TVT'))

        # --- Section 4: Display Checkboxes ---
        self.ax_toggles = self.fig_ctrl.add_axes([0.10, 0.28, 0.80, 0.11], facecolor='#f8f9fa')
        self.ax_toggles.set_title("Layer Visibility", fontsize=9, pad=4, loc='left', fontweight='bold')
        self.toggles = CheckButtons(self.ax_toggles, ['Show Geologist Mapping', 'Show Predicted Mapping'], [True, True])
        
        self.ax_lp_toggle = self.fig_ctrl.add_axes([0.10, 0.17, 0.80, 0.07], facecolor='#f8f9fa')
        self.ax_lp_toggle.set_title("Signal Processing", fontsize=9, pad=4, loc='left', fontweight='bold')
        self.lp_toggle = CheckButtons(self.ax_lp_toggle, ['Enable LP Filter'], [True])
        
        self.ax_legend_toggle = self.fig_ctrl.add_axes([0.10, 0.06, 0.80, 0.07], facecolor='#f8f9fa')
        self.ax_legend_toggle.set_title("Display Options", fontsize=9, pad=4, loc='left', fontweight='bold')
        self.legend_toggle = CheckButtons(self.ax_legend_toggle, ['Show Legends'], [True])

        # --- Bind Callbacks & Events ---
        self.ax_main.callbacks.connect('ylim_changed', self.sync_y_from_main)
        self.ax_right.callbacks.connect('ylim_changed', self.sync_y_from_right)
        
        self.fig_plot.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig_plot.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig_plot.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig_plot.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.slider_md.on_changed(self.update_plot)
        self.slider_m.on_changed(self.update_plot)
        self.slider_c.on_changed(self.update_plot)
        self.slider_cutoff.on_changed(self.update_plot)
        self.btn_commit.on_clicked(self.commit_chunk)
        self.btn_undo.on_clicked(self.undo_chunk)
        self.btn_reset.on_clicked(self.reset_all)
        
        self.radio_x.on_clicked(self.toggle_x_axis)
        self.radio_y.on_clicked(self.toggle_y_axis)
        self.toggles.on_clicked(self.toggle_visibility)
        self.lp_toggle.on_clicked(self.toggle_lp_filter)
        self.legend_toggle.on_clicked(self.toggle_legends)
        
        # --- Init Sync and Axis Limits ---
        self.fig_plot.canvas.draw()
        
        min_tvd, max_tvd = self.evalz['TVD'].min(), self.evalz['TVD'].max()
        buffer_tvd = abs(max_tvd - min_tvd) * 0.05
        if buffer_tvd == 0: buffer_tvd = 10
        self.ax_main.set_ylim(max_tvd + buffer_tvd, min_tvd - buffer_tvd) 

        self.initial_xlim = self.ax_top.get_xlim()
        self.initial_ylim = self.ax_main.get_ylim()
        
        self.update_plot(None)
        plt.show() 

    def toggle_x_axis(self, label):
        """Swaps the Main & Top panels between Measured Depth (MD) and True Horizontal Length (THL)"""
        self.top_axis_mode = label
        x_data = self.evalz[label]
        
        # Update base lines
        self.line_sensed_top.set_xdata(x_data)
        self.line_geo_top.set_xdata(x_data)
        self.line_traj_main.set_xdata(x_data)
        
        # Update committed chunks
        for i, chunk in enumerate(self.chunks):
            mask = (self.evalz['MD'] >= chunk['md_start']) & (self.evalz['MD'] <= chunk['md_end'])
            c_data = self.evalz[mask]
            self.history_lines_top[i].set_xdata(c_data[label])
            self.history_anchors_top[i].set_xdata([c_data[label].iloc[-1]])
            self.history_lines_main[i].set_xdata(c_data[label])
            
        # Update bounds and labels
        min_x, max_x = x_data.min(), x_data.max()
        self.ax_top.set_xlim(min_x - 50, max_x + 50)
        
        xlabel = "True Horizontal Length (THL)" if label == 'THL' else "Measured Depth (MD)"
        self.ax_main.set_xlabel(xlabel, fontsize=11)
        
        self.update_plot(None)

    def toggle_y_axis(self, label):
        """Swaps both the Main cross-section and Right correlation panel between TVD and TVT"""
        self.right_axis_mode = label
        
        if label == 'TVT':
            # In TVT Mode, update main graph y-axis and map curves to True Vertical Thickness
            self.ax_main.set_ylabel("True Vertical Thickness (TVT)", fontsize=11)
            self.line_geo_right.set_data(self.tw['GR'], self.tw['TVT'])
            self.line_sensed_right.set_label('Predicted HW TVT')
            self.line_traj_main.set_label('Predicted Trajectory')
            
            # Set Y bounds specifically for TVT range across both main and right panels
            min_y, max_y = self.tw['TVT'].min(), self.tw['TVT'].max()
            buffer_y = abs(max_y - min_y) * 0.05
            if buffer_y == 0: buffer_y = 10
            self.ax_main.set_ylim(max_y + buffer_y, min_y - buffer_y) # TVT increases downwards
            self.ax_right.set_ylim(max_y + buffer_y, min_y - buffer_y)
            
            self.ax_right.set_title("Correlation Panel (TVT)", fontsize=10)
            
        else: # TVD Mode
            self.ax_main.set_ylabel("True Vertical Depth (TVD)", fontsize=11)
            self.line_geo_right.set_data(self.truth_gr_full, self.evalz['TVD'])
            self.line_sensed_right.set_ydata(self.evalz['TVD'])
            
            min_tvd, max_tvd = self.evalz['TVD'].min(), self.evalz['TVD'].max()
            buffer_tvd = abs(max_tvd - min_tvd) * 0.05
            if buffer_tvd == 0: buffer_tvd = 10
            self.ax_main.set_ylim(max_tvd + buffer_tvd, min_tvd - buffer_tvd)
            
            self.ax_right.set_title("Correlation Panel (TVD)", fontsize=10)
            self.sync_y_from_main() # Snap back to Cross-Section's TVD bounds
            
            self.line_sensed_right.set_label('Sensed HW')
            self.line_traj_main.set_label('Trajectory')
            
        # Refresh legends to show updated labels context
        vis = self.legend_toggle.get_status()[0]
        if self.ax_main.get_legend(): 
            self.ax_main.legend(loc='upper right', fontsize=8).set_visible(vis)
        if self.ax_right.get_legend(): 
            self.ax_right.legend(loc='upper right', fontsize=8).set_visible(vis)
            
        # Update committed historical chunks for both main and right panels
        for i, chunk in enumerate(self.chunks):
            mask = (self.evalz['MD'] >= chunk['md_start']) & (self.evalz['MD'] <= chunk['md_end'])
            c_data = self.evalz[mask]
            if label == 'TVD':
                self.history_lines_main[i].set_ydata(c_data['TVD'])
                self.history_lines_right[i].set_ydata(c_data['TVD'])
            else: 
                norm_z_shift = c_data['norm_Z'] - c_data['norm_Z'].iloc[0]
                tvt_seg = chunk['m'] * (c_data['MD'] - chunk['md_start']) - norm_z_shift + chunk['tvt_0'] + chunk['c']
                self.history_lines_main[i].set_ydata(tvt_seg)
                self.history_lines_right[i].set_ydata(tvt_seg)
                
        self.update_plot(None)

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
        new_main_top = y_top + dy_dfig * (bbox_right.y1 - bbox_right.y1)
        self.ax_main.set_ylim(new_main_bottom, new_main_top)
        self._syncing = False

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None: return
        
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
        
        self.fig_plot.canvas.draw_idle()

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
        self.fig_plot.canvas.draw_idle()
        
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
        self.fig_plot.canvas.draw_idle()

    def toggle_legends(self, label):
        vis = self.legend_toggle.get_status()[0]
        if self.ax_top.get_legend(): self.ax_top.get_legend().set_visible(vis)
        if self.ax_main.get_legend(): self.ax_main.get_legend().set_visible(vis)
        if self.ax_right.get_legend(): self.ax_right.get_legend().set_visible(vis)
        self.fig_plot.canvas.draw_idle()

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

    def get_current_tvt_array(self, active_md_end, active_m, active_c):
        """Constructs a real-time TVT mapping array for the entire well based on active predictions"""
        tvt_arr = np.full(len(self.evalz), np.nan)
        
        # 1. Map established chunks
        for chunk in self.chunks:
            mask = (self.evalz['MD'] >= chunk['md_start']) & (self.evalz['MD'] <= chunk['md_end'])
            c_data = self.evalz[mask]
            norm_z_shift = c_data['norm_Z'] - c_data['norm_Z'].iloc[0]
            tvt_arr[mask] = chunk['m'] * (c_data['MD'] - chunk['md_start']) - norm_z_shift + chunk['tvt_0'] + chunk['c']
            
        # 2. Map active sliding prediction window
        active_mask = (self.evalz['MD'] >= self.current_md_start) & (self.evalz['MD'] <= active_md_end)
        if np.any(active_mask):
            c_data = self.evalz[active_mask]
            norm_z_shift = c_data['norm_Z'] - c_data['norm_Z'].iloc[0]
            tvt_arr[active_mask] = active_m * (c_data['MD'] - self.current_md_start) - norm_z_shift + self.current_tvt_0 + active_c
        
        # 3. Forecast unmapped future data based on last mapped trajectory
        unmapped_mask = (self.evalz['MD'] > active_md_end)
        if np.any(unmapped_mask):
            c_data = self.evalz[unmapped_mask]
            if np.any(active_mask):
                last_tvt = tvt_arr[active_mask][-1]
                last_norm_z = self.evalz[active_mask]['norm_Z'].iloc[-1]
            else:
                last_tvt = self.current_tvt_0
                last_norm_z = self.evalz['norm_Z'].iloc[0]
                
            tvt_arr[unmapped_mask] = -(c_data['norm_Z'] - last_norm_z) + last_tvt
            
        return tvt_arr

    def update_plot(self, val):
        active_md_end = self.slider_md.val
        active_m = self.slider_m.val
        active_c = self.slider_c.val
        active_cutoff = self.slider_cutoff.val
        
        if abs(active_cutoff - self.current_cutoff) > 1e-6:
            self.current_cutoff = active_cutoff
            self._apply_gr_filter()
        
        # --- Calculate History Metrics ---
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
                
                pred_gr_seg = np.interp(tvt_seg, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
                
                if 'TVT' in c_data.columns:
                    hist_true_tvt_list.append(c_data['TVT'])
                hist_pred_tvt_list.append(tvt_seg)
                hist_sensed_gr_list.append(c_data['GR'])
                hist_pred_gr_list.append(pred_gr_seg)
                
            if hist_true_tvt_list:
                full_hist_true_tvt = np.concatenate(hist_true_tvt_list)
                full_hist_pred_tvt = np.concatenate(hist_pred_tvt_list)
                hist_tvt_corr, hist_tvt_rmse = self._calculate_metrics(full_hist_true_tvt, full_hist_pred_tvt)
                
            full_hist_sensed_gr = np.concatenate(hist_sensed_gr_list)
            full_hist_pred_gr = np.concatenate(hist_pred_gr_list)
            hist_gr_corr, hist_gr_rmse = self._calculate_metrics(full_hist_sensed_gr, full_hist_pred_gr)
        
        # --- Evaluate Active Chunk ---
        valid_mask = (self.evalz['MD'] >= self.current_md_start) & (self.evalz['MD'] <= active_md_end)
        active_data = self.evalz[valid_mask]
        
        active_tvt_corr, active_tvt_rmse = 0.0, 0.0
        active_gr_corr, active_gr_rmse = 0.0, 0.0
        
        if len(active_data) > 0:
            active_norm_Z = active_data['norm_Z'] - active_data['norm_Z'].iloc[0]
            active_tvt = active_m * (active_data['MD'] - self.current_md_start) - active_norm_Z + self.current_tvt_0 + active_c
            
            active_pred_gr = np.interp(active_tvt, self.tw['TVT'], self.tw['GR'], left=np.nan, right=np.nan)
            
            # --- Update Active Lines mapped to Current Axis Settings ---
            active_x_data = active_data[self.top_axis_mode]
            self.line_pred_top.set_data(active_x_data, active_pred_gr)
            self.scatter_anchor_top.set_data([active_x_data.iloc[0]], [active_pred_gr[0]])
            
            if self.right_axis_mode == 'TVT':
                current_tvt_full = self.get_current_tvt_array(active_md_end, active_m, active_c)
                self.line_traj_main.set_ydata(current_tvt_full)
                self.line_pred_main.set_data(active_x_data, active_tvt)
                self.line_sensed_right.set_ydata(current_tvt_full)
                self.line_pred_right.set_data(active_pred_gr, active_tvt)
            else:
                self.line_traj_main.set_ydata(self.evalz['TVD'])
                self.line_pred_main.set_data(active_x_data, active_data['TVD'])
                self.line_sensed_right.set_ydata(self.evalz['TVD'])
                self.line_pred_right.set_data(active_pred_gr, active_data['TVD'])
            
            if 'TVT' in active_data.columns:
                active_tvt_corr, active_tvt_rmse = self._calculate_metrics(active_data['TVT'], active_tvt)
            active_gr_corr, active_gr_rmse = self._calculate_metrics(active_data['GR'], active_pred_gr)
            
        # Update Main Window Title with RMSE/Correlation Metrics
        title_str = "Interactive Geosteering Mode (Use Trackpad or Right-Click to Pan/Zoom)\n"
        if self.chunks:
            title_str += f"Hist    |   TVT: R={hist_tvt_corr:.3f}, RMSE={hist_tvt_rmse:.2f} ft   ||   GR: R={hist_gr_corr:.3f}, RMSE={hist_gr_rmse:.2f} API\n"
        else:
            title_str += "Hist    |   No chunks committed yet\n"
        if len(active_data) > 0:
            title_str += f"Active |   TVT: R={active_tvt_corr:.3f}, RMSE={active_tvt_rmse:.2f} ft   ||   GR: R={active_gr_corr:.3f}, RMSE={active_gr_rmse:.2f} API"
            
        self.fig_plot.suptitle(title_str, fontsize=11, fontweight='bold')
        self.fig_plot.canvas.draw_idle()
        
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
        
        # Plot lines respecting current axis mode
        x_data_active = active_data[self.top_axis_mode]
        l_top, = self.ax_top.plot(x_data_active, active_pred_gr, color='navy', alpha=0.7, linewidth=1.5, visible=vis)
        
        if self.right_axis_mode == 'TVT':
            l_main, = self.ax_main.plot(x_data_active, active_tvt, color='navy', alpha=0.7, linewidth=4, visible=vis)
            l_right, = self.ax_right.plot(active_pred_gr, active_tvt, color='navy', alpha=0.7, linewidth=1.5, visible=vis)
        else:
            l_main, = self.ax_main.plot(x_data_active, active_data['TVD'], color='navy', alpha=0.7, linewidth=4, visible=vis)
            l_right, = self.ax_right.plot(active_pred_gr, active_data['TVD'], color='navy', alpha=0.7, linewidth=1.5, visible=vis)
            
        anchor, = self.ax_top.plot([x_data_active.iloc[-1]], [active_pred_gr[-1]], 'bo', markersize=6, visible=vis)
        
        self.history_lines_top.append(l_top)
        self.history_lines_main.append(l_main)
        self.history_lines_right.append(l_right)
        self.history_anchors_top.append(anchor)
        
        self.chunks.append({
            'md_start': self.current_md_start,
            'md_end': active_md_end,
            'm': active_m,
            'c': active_c,
            'tvt_0': self.current_tvt_0
        })
        
        self.current_md_start = active_md_end
        self.current_tvt_0 = active_tvt.iloc[-1]
        
        # Reset sliders for next segment
        self.slider_md.valinit = min(self.current_md_start + 200, self.evalz['MD'].max())
        self.slider_md.reset()
        self.slider_md.valmin = self.current_md_start
        self.slider_md.ax.set_xlim(self.current_md_start, self.evalz['MD'].max())
        
        self.slider_m.reset()
        self.slider_c.reset()
        
        self.update_plot(None)
        
    def undo_chunk(self, event):
        if not self.chunks:
            return
            
        # Pop chunk data
        self.chunks.pop()
        
        # Remove graphical elements
        self.history_lines_top.pop().remove()
        self.history_lines_main.pop().remove()
        self.history_lines_right.pop().remove()
        self.history_anchors_top.pop().remove()
        
        if self.chunks:
            last_chunk = self.chunks[-1]
            self.current_md_start = last_chunk['md_end']
            
            # Recalculate tvt_0 based on the modified final state of the previous chunk
            mask = (self.evalz['MD'] >= last_chunk['md_start']) & (self.evalz['MD'] <= last_chunk['md_end'])
            c_data = self.evalz[mask]
            norm_z_shift = c_data['norm_Z'].iloc[-1] - c_data['norm_Z'].iloc[0]
            self.current_tvt_0 = last_chunk['m'] * (last_chunk['md_end'] - last_chunk['md_start']) - norm_z_shift + last_chunk['tvt_0'] + last_chunk['c']
        else:
            self.current_md_start = self.evalz['MD'].iloc[0]
            self.current_tvt_0 = self.evalz['TVT'].iloc[0] if 'TVT' in self.evalz.columns else 0.0
            
        # Reset sliders for next segment
        self.slider_md.valmin = self.current_md_start
        self.slider_md.ax.set_xlim(self.current_md_start, self.evalz['MD'].max())
        self.slider_md.valinit = min(self.current_md_start + 200, self.evalz['MD'].max())
        self.slider_md.reset()
        
        self.slider_m.reset()
        self.slider_c.reset()
        
        self.update_plot(None)
        
    def reset_all(self, event):
        self.chunks = []
        self.current_md_start = self.evalz['MD'].iloc[0]
        self.current_tvt_0 = self.evalz['TVT'].iloc[0] if 'TVT' in self.evalz.columns else 0.0
        
        for l in self.history_lines_top: l.remove()
        for l in self.history_lines_main: l.remove()
        for l in self.history_lines_right: l.remove()
        for a in self.history_anchors_top: a.remove()
        
        self.history_lines_top = []
        self.history_lines_main = []
        self.history_lines_right = []
        self.history_anchors_top = []
        
        self.slider_md.valmin = self.current_md_start
        self.slider_md.ax.set_xlim(self.current_md_start, self.evalz['MD'].max())
        self.slider_md.valinit = min(self.current_md_start + 200, self.evalz['MD'].max())
        self.slider_md.reset()
        self.slider_m.reset()
        self.slider_c.reset()
        
        self.ax_top.set_xlim(self.initial_xlim)
        self.ax_main.set_ylim(self.initial_ylim)
        self.update_plot(None)

if __name__ == '__main__':
    print("Launching Interactive Geosteering Mode...")
    print("Controls:")
    print("  - Use Mouse Middle/Right click to Pan.")
    print("  - Use Scroll Wheel to Zoom in/out at the cursor location.")
    print("  - Check out the brand-new Dynamic Axis Toggles.")
    simulator = GeosteeringSimulator(evalz, tw)