import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, colorchooser, ttk
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import rcParams
from matplotlib.colors import to_hex
from scipy.stats import zscore
from scipy.stats import ttest_ind
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib
import warnings
import webbrowser
import random

import matplotlib.pyplot as plt
import platform

# Detect OS
system = platform.system()

# Choose default Korean-safe font based on OS
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux or unknown
    plt.rcParams['font.family'] = 'DejaVu Sans'  # basic fallback, may not support Hangul

plt.rcParams['axes.unicode_minus'] = False  # fix minus sign issue


def compute_aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def find_best_poly_degree_aic(x, y, max_deg=30):
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return 1  # fallback to linear if too few points

    best_deg = 1
    best_aic = float('inf')
    max_allowed_deg = min(max_deg, len(x) - 1)

    for deg in range(1, max_allowed_deg + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                coeffs = np.polyfit(x, y, deg)
        except Exception:
            continue  # skip this degree if fitting fails

        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        residuals = y - y_pred
        rss = np.sum(residuals**2)

        aic = compute_aic(len(x), rss, deg + 1)

        # Optional: print for debugging
        #print(f"Degree {deg} → AIC: {aic:.2f}")

        if aic < best_aic:
            best_aic = aic
            best_deg = deg

    return best_deg
    
# Units mapping for axis labels
units = {
    'PM2.5_6sec': 'μg/m³', 'PM2.5_WAv': 'μg/m³', 'PM2.5_5min': 'μg/m³',
    'PM2.5_1hr': 'μg/m³', 'Temp': '°C', 'Humidity': '%',
    'PM10_WAv': 'μg/m³', 'PM10_5min': 'μg/m³',
    'CO2_6sec': 'ppm', 'CO2_WAv': 'ppm', 'CO2_1min': 'ppm',
    'Fan_Speed': 'RPM', 'TVOC': 'ppb'
}
plt.ioff()

sheet_names_list = {}
selected_sheets = {}
file_paths = {}
start_date_vars = {}
end_date_vars = {}
start_time_vars = {}
end_time_vars = {}
start_hour_vars = {}
start_min_vars  = {}
end_hour_vars   = {}
end_min_vars    = {}
original_xlim = None
original_ylim = None
annotations = []
scatter_plots = []
Avg_enabled = {}
Avg_color = []
date_ui_drawn = False
annotation_objects = []



def parse_file(path, sheet_name=None):
    ext = path.lower().split('.')[-1]

    def normalize_columns(df):
        df.columns = [col.strip().lower().replace(' ', '_').replace('¬∞', '') for col in df.columns]
        return df

    if ext == 'txt':
        pattern = re.compile(r"\{(.*?) SN:(.*?) \[(.*?)\]\}")
        with open(path, 'r') as f:
            rows = [m.groups() for line in f if (m := pattern.match(line.strip()))]
            if not rows:
                raise ValueError("No valid data lines found.")

        df = pd.DataFrame(rows, columns=['raw_dt', 'Serial', 'values'])

        # Split values column
        val_cols = df['values'].astype(str).str.split(',', expand=True)
        df.drop(columns=['values'], inplace=True)

        # Define master keys
        keys = ['PM2.5_6sec', 'PM2.5_WAv', 'PM2.5_5min', 'PM2.5_1hr',
                'Temp', 'Humidity', 'PM10_WAv', 'PM10_5min',
                'CO2_6sec', 'CO2_WAv', 'CO2_1min', 'Fan_Speed', 'TVOC',
                'Extra1', 'Extra2', 'Extra3']

        # Assign only the number of keys matching the values
        col_count = val_cols.shape[1]
        used_keys = keys[:col_count]
        val_cols.columns = used_keys

        # Merge with main DataFrame
        df = pd.concat([df, val_cols], axis=1)

        # Drop Extras if they exist
        for col in ['Extra1', 'Extra2', 'Extra3']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Convert numeric columns
        for col in used_keys:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].abs().max(skipna=True) <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        df['Timestamp'] = pd.to_datetime(df['raw_dt'], format='%y-%m-%d %H:%M:%S', errors='coerce')
        df.drop(columns=['raw_dt'], inplace=True)
        return df


    elif ext == 'csv':
        df = pd.read_csv(path, low_memory=False)

        # Clean and normalize column names
        df.columns = [col.strip() for col in df.columns]  # remove whitespace
        df.columns = [col.lower().replace('(', '').replace(')', '').replace('°c', 'c') for col in df.columns]

        # Drop INVALID column and duplicates
        df.drop(columns=[col for col in df.columns if 'invalid' in col], inplace=True, errors='ignore')
        df = df.loc[:, ~df.columns.duplicated()]

        if 'date' in df.columns and 'time' in df.columns:
            # Combine DATE and TIME columns
            df['Timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            try:
                df['TimeOnly'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
                df['TimeOnly'] = df['TimeOnly'].apply(
                    lambda t: datetime.combine(datetime.today(), t.time()) if pd.notna(t) else pd.NaT
                )
            except Exception:
                df['TimeOnly'] = pd.NaT

            df.drop(columns=['date', 'time'], inplace=True, errors='ignore')

        else:
            # Fallback: find first usable timestamp-ish column
            for col in df.columns:
                if any(x in col for x in ['timestamp', 'time', 'date']):
                    try:
                        df['Timestamp'] = pd.to_datetime(df[col], errors='coerce')
                        break
                    except Exception:
                        continue

            if 'Timestamp' not in df.columns:
                raise ValueError(f"No timestamp column found in {path}")

            df['TimeOnly'] = pd.NaT
            df.drop(columns=['date', 'time'], inplace=True, errors='ignore')

        df.dropna(subset=['Timestamp'], inplace=True)
        df.sort_values('Timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['elapsed_hours'] = np.arange(len(df)) * 6 / 3600

        return df


# --- GUI Setup ---
root = tk.Tk(); root.withdraw()
toolbar_frame = tk.Frame(root)
toolbar_frame.grid(row=2, column=0, columnspan=6, sticky='ew')
toolbar_frame.grid_propagate(False)
toolbar_frame.config(height=40)  # optional; set a fixed height
status = tk.StringVar(value='Ready')  # global variable
plot_frame = tk.Frame(root)
plot_frame.grid(row=3, column=0, columnspan=6, sticky='nsew')
global_start_date = tk.StringVar()
global_end_date = tk.StringVar()
sync_all_dates = tk.BooleanVar(value=False)
grid_interval = tk.IntVar(value=24)
num_sets = simpledialog.askinteger(
    "Number of Data Sets",         # title (positional)
    "How many datasets?",          # prompt (positional)
    minvalue=1,                    # keyword argument
    maxvalue=10                    # keyword argument
)




if num_sets is None:
    root.destroy(); exit()
root.deiconify(); root.title("Dynamic Multi–Dataset Plotter")
root.state('zoomed')
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(3, weight=1)

# Prepare colors
default_colors = [to_hex(c) for c in rcParams['axes.prop_cycle'].by_key()['color']]

# Globals
data_dfs   = [pd.DataFrame() for _ in range(num_sets)]
show_y     = [tk.BooleanVar(value=True) for _ in range(num_sets)]
show_y1    = [tk.BooleanVar(value=True) for _ in range(num_sets)]
filter_y   = tk.BooleanVar(value=False)
filter_y1  = tk.BooleanVar(value=False)
threshold  = tk.DoubleVar(value=3.0)
show_fit   = [tk.BooleanVar(value=False) for _ in range(num_sets)]
fit_degree = [tk.IntVar(value=2) for _ in range(num_sets)]
auto_fit = [tk.BooleanVar(value=True) for _ in range(num_sets)]
focus_mode = tk.BooleanVar(value=False)
show_legend = tk.BooleanVar(value=True)






    
# Special fixed colors for dataset 0
special_colors = {
    'y': '#0000FF',   
    'y1': '#FFA500',   
    'fit': '#ff0000',  
    'avg': '#00aa00'    
}

simple_colors_40 = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', '#ff4500', '#1e90ff',
    '#32cd32', '#ba55d3', '#ff69b4', '#cd5c5c', '#20b2aa', '#dda0dd', '#f0e68c', '#8b4513',
    '#2e8b57', '#b0c4de', '#ff6347', '#4682b4', '#6a5acd', '#00ced1', '#dc143c', '#d2691e'
]


# Indices for datasets 1 onward
random_indices = random.sample(range(1, 40), num_sets - 1)

y_color =    [tk.StringVar(value=special_colors['y'])]   + [tk.StringVar(value=simple_colors_40[i % 40]) for i in random_indices]
y1_color =   [tk.StringVar(value=special_colors['y1'])]  + [tk.StringVar(value=simple_colors_40[(i + 10) % 40]) for i in random_indices]
fit_color =  [tk.StringVar(value=special_colors['fit'])] + [tk.StringVar(value=simple_colors_40[(i + 20) % 40]) for i in random_indices]
Avg_color =  [tk.StringVar(value=special_colors['avg'])] + [tk.StringVar(value=simple_colors_40[(i + 30) % 40]) for i in random_indices]

x_var  = tk.StringVar(value='Timestamp')
y_var  = tk.StringVar(value='PM2.5_6sec')
y1_var = tk.StringVar(value='CO2_6sec')

current_fig = None
current_canvas = None

# Resize handler
root.bind('<Configure>', lambda e: _resize())
def _resize():
    global current_fig, current_canvas
    if current_fig and current_canvas:
        w, h = plot_frame.winfo_width(), plot_frame.winfo_height()
        dpi = current_fig.get_dpi()
        current_fig.set_size_inches(w/dpi, h/dpi)
        current_canvas.draw()

# UI Frames
frm = tk.Frame(root, padx=10, pady=10); frm.grid(row=0, column=0, sticky='ew')

# Color picker
def pick_color(i, axis):
    if axis == "y":
        new = colorchooser.askcolor(color=y_color[i].get())[1]
        if new:
            y_color[i].set(new)
            y_color_btns[i].config(bg=new)
    elif axis == "y1":
        new = colorchooser.askcolor(color=y1_color[i].get())[1]
        if new:
            y1_color[i].set(new)
            y1_color_btns[i].config(bg=new)
    elif axis == "fit":
        new = colorchooser.askcolor(color=fit_color[i].get())[1]
        if new:
            fit_color[i].set(new)
            fit_btns[i].config(bg=new)
    elif axis == "Avg":
        new = colorchooser.askcolor(color=Avg_color[i].get())[1]
        if new:
            Avg_color[i].set(new)
            Avg_btns[i].config(bg=new)


# Data Sets section
lf = tk.LabelFrame(frm, text="Loading & Graphing")
lf.grid(row=0, column=0, padx=(0, 10), sticky='nw')
sheet_vars = {}
sheet_menus = {}

columns_per_row = 5  # the number of columns per row you're using

for i in range(num_sets):
    col = i % columns_per_row
    row_offset = (i // columns_per_row) * 7  # adjust to the number of rows per set

    tk.Button(lf, text=f"Data Set {i+1}", command=lambda i=i: load_dataset(i)).grid(
        row=row_offset + 0, column=col, padx=2, sticky='ew'
    )

    sheet_vars[i] = tk.StringVar(value='Select sheet')
    sheet_menus[i] = ttk.OptionMenu(lf, sheet_vars[i], '')
    sheet_menus[i].grid(row=row_offset + 2, column=col, padx=2, sticky='w')
    sheet_menus[i].config(state='disabled')  # initially disabled

    tk.Checkbutton(lf, text=f"Y{i+1}", variable=show_y[i]).grid(
        row=row_offset + 3, column=col, sticky='w'
    )
    tk.Checkbutton(lf, text=f"Y1-{i+1}", variable=show_y1[i]).grid(
        row=row_offset + 4, column=col, sticky='w'
    )
    tk.Checkbutton(lf, text=f"Fit-{i+1}", variable=show_fit[i]).grid(
        row=row_offset + 5, column=col, sticky='w'
    )
    Avg_enabled[i] = tk.BooleanVar()
    tk.Checkbutton(lf, text=f"Hourly Avg-{i+1}", variable=Avg_enabled[i]).grid(
        row=row_offset + 6, column=col, sticky='w'
    )

for c in range(columns_per_row):
    lf.grid_columnconfigure(c, weight=1)

# Axes & Filters section
af = tk.LabelFrame(frm, text="Axes & Filters")
af.grid(row=0, column=2, columnspan=5, pady=5, sticky='nw')

# Polynomial Fit Options section
pf = tk.LabelFrame(frm, text="Colors")
pf.grid(row=0, column=10, columnspan=6, sticky='nw', pady=(5, 0), padx=10)

# Headers
tk.Label(pf, text="Data Set").grid(row=0, column=0)
tk.Label(pf, text="Y").grid(row=1, column=0)
tk.Label(pf, text="Y1").grid(row=2, column=0)
tk.Label(pf, text="Best Fit").grid(row=3, column=0)
tk.Label(pf, text="Hr Avg").grid(row=4, column=0)

for i in range(num_sets):


    y_color_btns = []
    y1_color_btns = []
    fit_btns = []
    Avg_btns = []
    fit_opts = []
    
    
    tk.Label(pf, text=f"{i+1}").grid(row=0, column=i+1)

    tk.Checkbutton(pf, variable=show_fit[i]).grid(row=1, column=i+1)
    
    
    b = tk.Button(pf, bg=y_color[i].get(), width=2, command=lambda i=i: pick_color(i,'y'))
    b.grid(row=1, column=i+1, padx=2); y_color_btns.append(b)
    b1= tk.Button(pf, bg=y1_color[i].get(), width=2, command=lambda i=i: pick_color(i,'y1'))
    b1.grid(row=2, column=i+1, padx=2); y1_color_btns.append(b1)

    b_fit = tk.Button(pf, bg=fit_color[i].get(), width=2,command=lambda i=i: pick_color(i, "fit"))
    b_fit.grid(row=3, column=i+1); fit_btns.append(b_fit)

    b_av = tk.Button(pf, bg=Avg_color[i].get(), width=2, command=lambda i=i: pick_color(i, "Avg"))
    b_av.grid(row=4, column=i+1); Avg_btns.append(b_av)




# X axis selector
tk.Label(af, text='X:').grid(row=0, column=0)
x_opt = tk.OptionMenu(af, x_var, '')
x_opt.grid(row=0, column=1)

# Y axis selector
tk.Label(af, text='Y:').grid(row=0, column=2)
y_opt = tk.OptionMenu(af, y_var, '')
y_opt.grid(row=0, column=3)

# Y1 axis selector
tk.Label(af, text='Y1:').grid(row=1, column=2)
y1_opt = tk.OptionMenu(af, y1_var, '')
y1_opt.grid(row=1, column=3)

tk.Label(af, text='Z-Threshold').grid(row=2, column=0)
tk.Spinbox(af, from_=0, to=10, increment=0.1, textvariable=threshold, width=5).grid(row=2, column=1)
tk.Checkbutton(af, text='Filter Y', variable=filter_y).grid(row=2, column=2)
tk.Checkbutton(af, text='Filter Y1', variable=filter_y1).grid(row=2, column=3)
bottom_frame = tk.Frame(root)
bottom_frame.grid(row=99, column=0, columnspan=7, pady=20, sticky='we') 

status_bar = tk.Label(bottom_frame, textvariable=status, relief='sunken', anchor='w')
status_bar.grid(row=1, column=0, columnspan=6, sticky='we', pady=(10, 0))

# List of UI frames to hide in focus mode (except control_frame and plot area)
hideable_frames = [lf, af, pf, bottom_frame]  

def toggle_focus_mode():
    if focus_mode.get():
        # Turn OFF focus mode → show all frames again
        for f in hideable_frames:
            f.grid()
        focus_mode.set(False)
        focus_btn.config(text="Hide UI")
    else:
        # Turn ON focus mode → hide all UI frames except controls & plot
        for f in hideable_frames:
            f.grid_remove()
        focus_mode.set(True)
        focus_btn.config(text="Show UI")

def toggle_legend():
    if show_legend.get():
        show_legend.set(False)
        legend_btn.config(text="Show Legend")
    else:
        show_legend.set(True)
        legend_btn.config(text="Hide Legend")
    update_legend()

def update_legend():
    if current_fig and current_canvas:
        ax = current_fig.axes[0]
        ax2 = ax.twinx() if len(current_fig.axes) > 1 else None
        ax.legend_.remove() if ax.legend_ else None
        if show_legend.get():
            h, l = ax.get_legend_handles_labels()
            if ax2:
                h2, l2 = ax2.get_legend_handles_labels()
                h += h2
                l += l2
            ax.legend(h, l)
        current_canvas.draw_idle()

from scipy.stats import ttest_ind, levene

def run_ttest():
    if len(data_dfs) < 2:
        messagebox.showwarning("Need Two Datasets", "At least two datasets must be loaded to run a t-test.")
        return

    dataset_indices = [i for i in range(len(data_dfs)) if not data_dfs[i].empty]
    if len(dataset_indices) < 2:
        messagebox.showwarning("Insufficient Data", "Two non-empty datasets are required.")
        return

    ycol = y_var.get()
    d1, d2 = data_dfs[dataset_indices[0]], data_dfs[dataset_indices[1]]

    y1 = pd.to_numeric(d1[ycol], errors='coerce').dropna()
    y2 = pd.to_numeric(d2[ycol], errors='coerce').dropna()

    if filter_y.get():
        y1 = y1[np.abs(zscore(y1)) <= threshold.get()]
        y2 = y2[np.abs(zscore(y2)) <= threshold.get()]

    if len(y1) < 2 or len(y2) < 2:
        messagebox.showwarning("Insufficient Data", "Each group must have at least two valid data points.")
        return

    # --- Variance equality check (Levene’s test) ---
    stat_var, p_var = levene(y1, y2)
    equal_var = p_var > 0.05  # If p > 0.05 → assume equal variances

    # --- t-test (Student or Welch) ---
    t_stat, p_val = ttest_ind(y1, y2, equal_var=equal_var)
    test_type = "Student’s t-test (equal variances)" if equal_var else "Welch’s t-test (unequal variances)"

    # --- Show results ---
    messagebox.showinfo("T-Test Result",
                        f"Comparing: Dataset {dataset_indices[0] + 1} vs Dataset {dataset_indices[1] + 1}\n"
                        f"Variable: {ycol}\n\n"
                        f"Variance equality test (Levene’s) p = {p_var:.4f}\n"
                        f"→ Using {test_type}\n\n"
                        f"Mean 1: {y1.mean():.2f}  (n={len(y1)})\n"
                        f"Mean 2: {y2.mean():.2f}  (n={len(y2)})\n\n"
                        f"t = {t_stat:.3f}\n"
                        f"p = {p_val:.4f} (two-sided)")
    

def export_filtered_data():
    from tkinter import filedialog
    export_dir = filedialog.askdirectory(title="Choose folder to save filtered datasets")
    if not export_dir:
        return

    for i, df in enumerate(data_dfs):
        if df.empty or 'Timestamp' not in df.columns:
            continue


        try:
            df_copy = df.copy()
            if sync_all_dates.get():
                start = pd.to_datetime(global_start_date.get() + ' ' + start_time_vars[i].get())
                end = pd.to_datetime(global_end_date.get() + ' ' + end_time_vars[i].get())
            else:
                start = pd.to_datetime(start_date_vars[i].get() + ' ' + start_time_vars[i].get())
                end = pd.to_datetime(end_date_vars[i].get() + ' ' + end_time_vars[i].get())

            filtered_df = df_copy[(df_copy['Timestamp'] >= start) & (df_copy['Timestamp'] <= end)]

            if filtered_df.empty:
                continue

            # Build filename
            base = file_paths.get(i, f"dataset_{i+1}").split('/')[-1].split('\\')[-1].split('.')[0]
            sheet = f"_{selected_sheets[i]}" if i in selected_sheets else ""
            # Format date range
            start_str = start.strftime('%Y%m%d')
            end_str = end.strftime('%Y%m%d')
            out_name = f"{base}{sheet}_{start_str}~{end_str}_filtered.csv"

            filtered_df.to_csv(f"{export_dir}/{out_name}", index=False)
        except Exception as e:
            print(f"Export failed for dataset {i+1}: {e}")

    messagebox.showinfo("Export Complete", "Filtered datasets have been saved.")

def export_hourly_averages():
    export_dir = filedialog.askdirectory(title="Choose folder to save hourly averages")
    if not export_dir:
        return

    for i, df in enumerate(data_dfs):
        if df.empty or 'Timestamp' not in df.columns or not Avg_enabled[i].get():
            continue

        try:
            df = df.dropna(subset=['Timestamp', y_var.get()])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

            # Resample to hourly average
            avg_df = df.resample('1H', on='Timestamp')[y_var.get()].mean().reset_index()
            avg_df = avg_df.dropna()

            if avg_df.empty:
                continue

            # Build filename
            base = file_paths.get(i, f"dataset_{i+1}").split('/')[-1].split('\\')[-1].split('.')[0]
            sheet = f"_{selected_sheets[i]}" if i in selected_sheets else ""
            # Try to extract start/end dates from filtered data
            try:
                start_str = df['Timestamp'].min().strftime('%Y%m%d')
                end_str = df['Timestamp'].max().strftime('%Y%m%d')
                date_part = f"_{start_str}~{end_str}"
            except:
                date_part = ""

            out_name = f"{base}{sheet}{date_part}_hourly_avg.csv"

            avg_df.to_csv(f"{export_dir}/{out_name}", index=False)
        except Exception as e:
            print(f"Hourly average export failed for dataset {i+1}: {e}")

    messagebox.showinfo("Export Complete", "Hourly average datasets have been saved.")

def show_descriptives():
    ycol = y_var.get()

    msg = ""
    for i, df in enumerate(data_dfs):
        if df.empty or ycol not in df.columns:
            continue

        series = pd.to_numeric(df[ycol], errors='coerce').dropna()

        if filter_y.get():
            series = series[np.abs(zscore(series)) <= threshold.get()]

        if series.empty:
            continue

        desc = series.describe()
        msg += (f"Dataset {i+1} - {ycol}:\n"
                f"Count: {int(desc['count'])}\n"
                f"Mean: {desc['mean']:.2f}\n"
                f"Std: {desc['std']:.2f}\n"
                f"Min: {desc['min']:.2f}\n"
                f"25%: {desc['25%']:.2f}\n"
                f"50% (Median): {desc['50%']:.2f}\n"
                f"75%: {desc['75%']:.2f}\n"
                f"Max: {desc['max']:.2f}\n\n")

    if msg == "":
        msg = "No data available for descriptives."
    messagebox.showinfo("Descriptive Statistics", msg)

    
# T-Test button added to row 0, column 6 to align with top-right placement
ttest_btn = tk.Button(af, text="Run T-Test", command=run_ttest)
ttest_btn.grid(row=1, column=6, padx=10, pady=5, sticky='w')
export_btn = tk.Button(af, text="Export Filtered Data", command=export_filtered_data)
export_btn.grid(row=2, column=6, padx=10, pady=5, sticky='w')
desc_btn = tk.Button(af, text="Descriptives", command=show_descriptives)
desc_btn.grid(row=0, column=6, padx=10, pady=5, sticky='w')
export_avg_btn = tk.Button(af, text="Export Hourly Averages", command=export_hourly_averages)
export_avg_btn.grid(row=3, column=6, padx=10, pady=5, sticky='w')

# Plot area
plot_frame.grid(row=3, column=0, columnspan=6, sticky='nsew')

# Handlers
def reload_sheet_gui(i, sheet):
    try:
        selected_sheets[i] = sheet
        df = parse_file(file_paths[i], sheet_name=sheet)
        data_dfs[i] = df
        status.set(f"Reloaded set {i+1} with sheet '{sheet}' ({len(df)} rows)")
        if i == 0:
            refresh_axes()
    except Exception as e:
        messagebox.showerror("Sheet Load Error", str(e))


def load_dataset(i):
    path = filedialog.askopenfilename(filetypes=[
        ('All supported', '*.txt *.csv *.xlsx'),
        ('Text files', '*.txt'),
        ('CSV files', '*.csv'),
        ('Excel files', '*.xlsx')
    ])
    if not path:
        return

    ext = path.lower().split('.')[-1]
    try:
        if ext == 'xlsx':
            xls = pd.ExcelFile(path)
            sheet_names_list[i] = xls.sheet_names
            selected_sheets[i] = xls.sheet_names[0]  # default
            file_paths[i] = path

            # GUI-driven sheet dropdown
            sheet_vars[i].set(xls.sheet_names[0])
            menu = sheet_menus[i]['menu']
            menu.delete(0, 'end')
            for s in sheet_names_list[i]:
                menu.add_command(label=s, command=lambda v=s, i=i: reload_sheet_gui(i, v))
            sheet_menus[i].config(state='normal')

            df = parse_file(path, sheet_name=sheet_vars[i].get())
        else:
            df = parse_file(path)
            file_paths[i] = path

        data_dfs[i] = df

        # Auto-select Y and Y1 after loading dataset 0
        if i == 0 and not df.empty:
            if any(c.lower().startswith('pm2') for c in df.columns):
                y_var.set([c for c in df.columns if c.lower().startswith('pm2')][0])
            if any('co2' in c.lower() for c in df.columns):
                y1_var.set([c for c in df.columns if 'co2' in c.lower()][0])
            refresh_axes()

        status.set(f"Loaded set {i+1} ({len(df)} rows)")
        plot_btn.config(state='normal')

    except Exception as e:
        messagebox.showerror("Error", str(e))



        
def select_sheet(i):
    if i not in file_paths or i not in sheet_names_list:
        messagebox.showwarning("No Excel file", "No Excel file loaded for this slot.")
        return

    # Create a dropdown-style selection window
    win = tk.Toplevel()
    win.title(f"Select Sheet for Dataset {i+1}")
    tk.Label(win, text="Choose sheet:").pack(pady=5)

    selected = tk.StringVar(value=selected_sheets.get(i, sheet_names_list[i][0]))
    sheet_dropdown = ttk.OptionMenu(win, selected, selected.get(), *sheet_names_list[i])
    sheet_dropdown.pack(pady=5, padx=10)

    def apply_selection():
        sheet = selected.get()
        if sheet not in sheet_names_list[i]:
            messagebox.showerror("Invalid Sheet", "That sheet name does not exist in the file.")
            return
        try:
            selected_sheets[i] = sheet
            df = parse_file(file_paths[i], sheet_name=sheet)
            data_dfs[i] = df
            status.set(f"Reloaded set {i+1} with sheet '{sheet}' ({len(df)} rows)")
            if i == 0:
                refresh_axes()
            win.destroy()
        except Exception as e:
            messagebox.showerror("Sheet Load Error", str(e))

    tk.Button(win, text="Load Sheet", command=apply_selection).pack(pady=10)

def refresh_axes():
    df = data_dfs[0]
    if df.empty: return
    cols = [c for c in df.columns if c not in ('raw_dt','Serial','values')]
    opts = ['Timestamp','elapsed_hours']+cols
    for widget,var in [(af.winfo_children()[1], x_var), (af.winfo_children()[3], y_var), (af.winfo_children()[5], y1_var)]:
        menu = widget['menu']
        menu.delete(0, 'end')
        for c in opts:
            menu.add_command(label=c, command=lambda v=c, m=var: m.set(v))

def get_all_dates():
    all_dates = []
    for df in data_dfs:
        if not df.empty and 'Timestamp' in df.columns:
            dates = pd.to_datetime(df['Timestamp'], errors='coerce').dropna().dt.date
            all_dates.extend(dates)
    return sorted(set(all_dates))


def on_pick(event):


    artist = event.artist
    axis = artist.axes
    ind = event.ind[0]

    for sc, i, label_type, col_name in scatter_plots:
        if sc == artist:
            df = data_dfs[i]
            x = pd.to_datetime(df['Timestamp'], errors='coerce')
            y = pd.to_numeric(df[col_name], errors='coerce')
            mask = ~x.isna() & ~y.isna()
            x = x[mask]
            y = y[mask]
            # Defensive length check
            if 0 <= ind < len(x):
                ann = axis.annotate(
                    f"Dataset {i+1}\n{col_name}\n{str(x.iloc[ind])}\n{y.iloc[ind]:.3f}",
                    xy=(x.iloc[ind], y.iloc[ind]),
                    xytext=(10, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->")
                )
                annotation_objects.append(ann)
                current_canvas.draw()



def clear_annotations():
    global annotation_objects
    print("Clearing", len(annotation_objects), "annotations")  # debug line
    for ann in annotation_objects:
        ann.remove()
    annotation_objects.clear()
    if current_fig:
        current_fig.canvas.draw()

def clear_all_datasets():
    confirm = messagebox.askyesno(
        title="Clear All Datasets?",
        message="This will remove all loaded data, clear plots, and reset filters.\nAre you sure you want to continue?"
    )
    if not confirm:
        return 
        
    global data_dfs, file_paths, selected_sheets, sheet_names_list
    global start_date_vars, end_date_vars, start_time_vars, end_time_vars
    global original_xlim, original_ylim, current_canvas, current_fig
    global date_ui_drawn

    # Clear core data
    data_dfs = [pd.DataFrame() for _ in range(num_sets)]
    file_paths.clear()
    selected_sheets.clear()
    sheet_names_list.clear()

    # Clear date/time filter variables
    start_date_vars.clear()
    end_date_vars.clear()
    start_time_vars.clear()
    end_time_vars.clear()

    # Reset zoom
    original_xlim = None
    original_ylim = None

    # Clear plot frame
    for w in plot_frame.winfo_children():
        w.destroy()

    # Clear bottom UI (date controls)
    for w in bottom_frame.winfo_children():
        w.destroy()

    # Reset current figure/canvas
    current_fig = None
    current_canvas = None

    # Reset flags
    globals()["date_ui_drawn"] = False
    globals()["traces_added"] = False

    # Reset status
    status.set("All datasets cleared. Ready to load new data.")

def add_polynomial_fit(ax, X, Y, label, degree, is_timestamp, color):
    try:
        # Convert datetime to float if X is time
        if is_timestamp:
            X_numeric = mdates.date2num(X)
        else:
            X_numeric = pd.to_numeric(X, errors='coerce')

        mask = ~np.isnan(X_numeric) & ~np.isnan(Y)
        X_fit = X_numeric[mask]
        Y_fit = Y[mask]

        if len(X_fit) < degree + 1:
            return  # not enough points for this degree

        coeffs = np.polyfit(X_fit, Y_fit, degree)
        poly = np.poly1d(coeffs)

        X_sorted = np.sort(X_fit)
        Y_smooth = poly(X_sorted)

        if is_timestamp:
            X_sorted = mdates.num2date(X_sorted)

        ax.plot(X_sorted, Y_smooth, color=color, linestyle='-',
                linewidth=2, label=f"{label} (Poly Deg {degree})")
    except Exception as e:
        print(f"Fit error: {e}")

def apply_time_grid(ax):
    # Only for Timestamp x-axis
    if x_var.get() != 'Timestamp' or not ax:
        return
    try:
        h = int(grid_interval.get())
    except Exception:
        h = 1
    h = max(1, h)

    locator = mdates.HourLocator(interval=h)
    # Show date + hour when < 24h steps, otherwise just date
    fmt = '%Y-%m-%d\n%H:%M' if h < 24 else '%Y-%m-%d'
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

def plot_all():
    global current_fig, current_canvas, original_xlim, original_ylim, canvas, toolbar, traces_added, date_ui_drawn
    scatter_plots.clear()
    annotation_objects.clear()

    if data_dfs[0].empty:
        messagebox.showwarning('No data', 'Load dataset 1')
        return

    fig, ax = plt.subplots()
    ax.format_coord = lambda x, y: ""
    ax2 = ax.twinx()
    ax2.format_coord = lambda x, y: ""
    current_fig = fig

    thr = threshold.get()
    filtered_x_all = []

    for i, df in enumerate(data_dfs):
        if df.empty or 'Timestamp' not in df.columns:
            continue

        # Ensure time filter variables exist
        if i not in start_date_vars:
            start_date_vars[i] = tk.StringVar()
            end_date_vars[i]   = tk.StringVar()
            start_time_vars[i] = tk.StringVar()  # legacy; unused after HH/MM split
            end_time_vars[i]   = tk.StringVar()  # legacy; unused after HH/MM split

        # Ensure hour/minute vars exist
        if i not in start_hour_vars: start_hour_vars[i] = tk.StringVar()
        if i not in start_min_vars:  start_min_vars[i]  = tk.StringVar()
        if i not in end_hour_vars:   end_hour_vars[i]   = tk.StringVar()
        if i not in end_min_vars:    end_min_vars[i]    = tk.StringVar()

        # Default date/time dropdown values
        try:
            ts = pd.to_datetime(df['Timestamp'], errors='coerce').dropna()
            date_options = sorted(ts.dt.date.unique())
            if not start_date_vars[i].get() and date_options:
                start_date_vars[i].set(str(date_options[0]))
            if not end_date_vars[i].get() and date_options:
                end_date_vars[i].set(str(date_options[-1]))

            if not ts.empty:
                first_ts, last_ts = ts.iloc[0], ts.iloc[-1]
                if not start_hour_vars[i].get(): start_hour_vars[i].set(f"{first_ts.hour:02d}")
                if not start_min_vars[i].get():  start_min_vars[i].set(f"{first_ts.minute:02d}")
                if not end_hour_vars[i].get():   end_hour_vars[i].set(f"{last_ts.hour:02d}")
                if not end_min_vars[i].get():    end_min_vars[i].set(f"{last_ts.minute:02d}")
        except Exception as e:
            print(f"Dropdown init error (Dataset {i+1}):", e)

        # Apply time filtering
        try:
            if sync_all_dates.get():
                start_str = f"{global_start_date.get()} {start_hour_vars[i].get()}:{start_min_vars[i].get()}"
                end_str   = f"{global_end_date.get()} {end_hour_vars[i].get()}:{end_min_vars[i].get()}"
            else:
                start_str = f"{start_date_vars[i].get()} {start_hour_vars[i].get()}:{start_min_vars[i].get()}"
                end_str   = f"{end_date_vars[i].get()} {end_hour_vars[i].get()}:{end_min_vars[i].get()}"

            start = pd.to_datetime(start_str, errors='coerce')
            end   = pd.to_datetime(end_str, errors='coerce')

            df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
        except Exception as e:
            print(f"Time filter error (Dataset {i+1}):", e)

        if df.empty:
            continue

        # X values
        if x_var.get() == 'Timestamp':
            X = pd.to_datetime(df['Timestamp'], errors='coerce')
            filtered_x_all.extend(X.dropna())
            is_ts = True
        else:
            X = pd.to_numeric(df[x_var.get()], errors='coerce')
            filtered_x_all.extend(pd.Series(X).dropna())
            is_ts = False

        # Label prefix
        label_prefix = file_paths.get(i, f"Dataset {i+1}").split('/')[-1].split('\\')[-1]
        if i in selected_sheets:
            label_prefix += f" ({selected_sheets[i]})"

        # --- Y1 scatter ---
        if show_y1[i].get() and y1_var.get() in df.columns:
            y1 = pd.to_numeric(df[y1_var.get()], errors='coerce')
            mask_y1 = (abs(zscore(y1.ffill())) <= thr) if filter_y1.get() and y1.std() else ~y1.isna()
            sc1 = ax2.scatter(X[mask_y1], y1[mask_y1], color=y1_color[i].get(),
                              label=f"{label_prefix} - {y1_var.get()}", s=10, picker=True)
            scatter_plots.append((sc1, i, 'y1', y1_var.get()))

        # --- Y scatter + (optional) polynomial fit ---
        if show_y[i].get() and y_var.get() in df.columns:
            y = pd.to_numeric(df[y_var.get()], errors='coerce')
            mask_y = (abs(zscore(y.ffill())) <= thr) if filter_y.get() and y.std() else ~y.isna()

            sc0 = ax.scatter(X[mask_y], y[mask_y], color=y_color[i].get(),
                             label=f"{label_prefix} - {y_var.get()}", s=10, picker=True)
            scatter_plots.append((sc0, i, 'y', y_var.get()))

            if show_fit[i].get():
                x_vals = (mdates.date2num(X[mask_y]) if is_ts
                          else pd.to_numeric(X[mask_y], errors='coerce'))
                y_vals = y[mask_y]

                valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                if np.sum(valid) >= 3:
                    best_deg = find_best_poly_degree_aic(x_vals[valid], y_vals[valid], max_deg=30)
                    fit_degree[i].set(best_deg)

                    X_for_fit = X[mask_y][valid]
                    Y_for_fit = y[mask_y][valid]

                    add_polynomial_fit(
                        ax=ax,
                        X=X_for_fit,
                        Y=Y_for_fit,
                        label=f"{label_prefix} - {y_var.get()}",
                        degree=best_deg,
                        is_timestamp=is_ts,
                        color=fit_color[i].get()
                    )

        # --- Hourly average (if toggled) ---
        if Avg_enabled.get(i, tk.BooleanVar()).get() and y_var.get() in df.columns:
            df_avg = df.dropna(subset=['Timestamp', y_var.get()]).copy()
            df_avg['Timestamp'] = pd.to_datetime(df_avg['Timestamp'], errors='coerce')
            Avg = df_avg.resample('1h', on='Timestamp')[y_var.get()].mean().dropna()
            if not Avg.empty:
                ax.plot(Avg.index, Avg.values, 'o-', color=Avg_color[i].get(),
                        label=f"{label_prefix} - Hourly Avg")

    # Axes labels
    ax.set_xlabel(x_var.get())
    ax.set_ylabel(f"{y_var.get()} ({units.get(y_var.get(), '')})")
    if any(show_y1[j].get() for j in range(len(data_dfs))):
        ax2.set_ylabel(f"{y1_var.get()} ({units.get(y1_var.get(), '')})")

    ax.grid(True)

    # Dynamic x-limits based on filtered data
    if filtered_x_all:
        ax.set_xlim(min(filtered_x_all), max(filtered_x_all))

    # Format x-axis
    if x_var.get() == 'Timestamp':
        apply_time_grid(ax)
        fig.autofmt_xdate(rotation=45, ha='right')
    else:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))

    fig.tight_layout()

    # Save axis limits for zoom reset
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    # Embed canvas
    for w in plot_frame.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky='nsew')
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    # Refresh toolbar
    for widget in toolbar_frame.winfo_children():
        widget.destroy()
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.grid(row=0, column=0, sticky='ew')

    current_canvas = canvas
    current_fig = fig

    # Events
    canvas.mpl_connect('pick_event', on_pick)
    canvas.mpl_connect('button_release_event', on_zoom_release)

    # Only attach trace callbacks once
    if not globals().get("traces_added", False):
        add_trace_callbacks()
        globals()["traces_added"] = True

    # Build bottom UI (global + per-dataset date/time controls) once
    if not globals().get("date_ui_drawn", False):
        global_frame = tk.LabelFrame(bottom_frame, text="Global Controls", padx=10, pady=5)
        global_frame.grid(row=0, column=0, columnspan=3, sticky='sw', pady=10)

        tk.Label(global_frame, text="Global Start:").grid(row=0, column=0, sticky='w')
        tk.OptionMenu(global_frame, global_start_date, *get_all_dates()).grid(row=0, column=1, sticky='w')

        tk.Label(global_frame, text="Global End:").grid(row=0, column=2, sticky='w')
        tk.OptionMenu(global_frame, global_end_date, *get_all_dates()).grid(row=0, column=3, sticky='w')

        tk.Checkbutton(global_frame, text="Sync All Dates", variable=sync_all_dates).grid(row=0, column=4, padx=10, sticky='w')

        tk.Label(global_frame, text="Grid Interval (hours):").grid(row=0, column=5, sticky='w')
        tk.OptionMenu(global_frame, grid_interval, *list(range(1, 25))).grid(row=0, column=6, padx=5, sticky='w')

        # Per-dataset controls
        for i in range(len(data_dfs)):
            if i in start_date_vars and not data_dfs[i].empty:
                dataset_time_frame = tk.LabelFrame(bottom_frame, text=f"Dataset {i+1} Controls", padx=10, pady=5)
                dataset_time_frame.grid(row=0, column=i+3, columnspan=1, sticky='sw', padx=10, pady=5)

                dates = sorted(set(pd.to_datetime(data_dfs[i]['Timestamp'], errors='coerce').dropna().dt.date))
                tk.Label(dataset_time_frame, text="Start Date:").grid(row=0, column=0)
                tk.OptionMenu(dataset_time_frame, start_date_vars[i], *dates).grid(row=0, column=1, sticky='w')

                tk.Label(dataset_time_frame, text="End Date:").grid(row=1, column=0)
                tk.OptionMenu(dataset_time_frame, end_date_vars[i], *dates).grid(row=1, column=1, sticky='w')

                # Time HH:MM selectors
                hours = [f"{h:02d}" for h in range(24)]
                mins  = [f"{m:02d}" for m in range(60)]

                tk.Label(dataset_time_frame, text="Start Time:").grid(row=2, column=0)
                tk.OptionMenu(dataset_time_frame, start_hour_vars[i], *hours).grid(row=2, column=1, sticky='w')
                tk.Label(dataset_time_frame, text=":").grid(row=2, column=2, sticky='w')
                tk.OptionMenu(dataset_time_frame, start_min_vars[i], *mins).grid(row=2, column=3, sticky='w')

                tk.Label(dataset_time_frame, text="End Time:").grid(row=3, column=0)
                tk.OptionMenu(dataset_time_frame, end_hour_vars[i], *hours).grid(row=3, column=1, sticky='w')
                tk.Label(dataset_time_frame, text=":").grid(row=3, column=2, sticky='w')
                tk.OptionMenu(dataset_time_frame, end_min_vars[i], *mins).grid(row=3, column=3, sticky='w')

        globals()["date_ui_drawn"] = True

    current_canvas.draw()
    status.set("Plot updated.")



def add_trace_callbacks():
    for var_dict in [start_date_vars, end_date_vars, start_hour_vars, start_min_vars, end_hour_vars, end_min_vars]:
        for i in var_dict:
            var = var_dict[i]
            def cb(*args, i=i):
                if suspend_traces:  # ignore changes while we’re initializing defaults
                    return
                schedule_plot()
            var.trace_add("write", cb)


def help_btn():
    url = "https://hward05.github.io/help-page-/"  # Replace with your actual help/documentation URL
    webbrowser.open(url)
    status.set('Plot ready')
    
def on_zoom_release(event):
    
    if current_canvas:
        ax = current_fig.axes[0]
        if x_var.get() == 'Timestamp':
            locator = mdates.AutoDateLocator(minticks=10, maxticks=12)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        current_canvas.draw_idle()

    if not current_fig or not current_canvas:
        return
    ax = current_fig.axes[0]
    xlim = ax.get_xlim()
    
    for i, df in enumerate(data_dfs):
        if df.empty or not show_fit[i].get():
            continue

        # Get the current X and Y
        if x_var.get() == 'Timestamp':
            X = mdates.date2num(df['Timestamp'])
            is_ts = True
        else:
            X = pd.to_numeric(df[x_var.get()], errors='coerce')
            is_ts = False

        Y = pd.to_numeric(df[y_var.get()], errors='coerce')

        # Mask by zoomed-in range
        mask = ~np.isnan(X) & ~np.isnan(Y) & (X >= xlim[0]) & (X <= xlim[1])
        X_fit = X[mask]
        Y_fit = Y[mask]

        if len(X_fit) < 3:
            continue  # not enough points

        # Remove old fit lines with matching label
        label_prefix = file_paths.get(i, f"Dataset {i+1}").split('/')[-1].split('\\')[-1]
        if i in selected_sheets:
            label_prefix += f" ({selected_sheets[i]})"
        fit_label = f"{label_prefix} - {y_var.get()} (Poly Deg"

        # Remove old fit line
        for line in ax.lines[:]:
            if line.get_label().startswith(fit_label):
                line.remove()

        # Recalculate degree
        degree = (find_best_poly_degree_aic(X_fit, Y_fit, max_deg=30)
                  if auto_fit[i].get() else fit_degree[i].get())

        fit_degree[i].set(degree)

        # Refit and redraw
        coeffs = np.polyfit(X_fit, Y_fit, degree)
        poly = np.poly1d(coeffs)
        X_sorted = np.sort(X_fit)
        Y_smooth = poly(X_sorted)

        if is_ts:
            X_sorted = mdates.num2date(X_sorted)

        ax.plot(X_sorted, Y_smooth, color=fit_color[i].get(), linewidth=2,
                label=f"{label_prefix} - {y_var.get()} (Poly Deg {degree})")

    # Update legend only if toggle is enabled
        if show_legend.get():
            ax.legend(loc='upper left', fontsize='small')
        else:
            ax.get_legend().remove() if ax.get_legend() else None


# Frame for Plot controls
control_frame = tk.LabelFrame(frm, text="General Controls", padx=5, pady=5)
control_frame.grid(row=0, column=1, sticky='n', pady=5, padx=5)

# Place this after all other frames/grids are set up—right before 'root.mainloop()'
# After defining control_frame or near bottom UI layout:




n_cols = 3  

for col in range(n_cols):
    control_frame.grid_columnconfigure(col, weight=1)

plot_btn = tk.Button(control_frame, text='Plot', state='disabled', command=plot_all)
clear_btn = tk.Button(control_frame, text='Clear', command=clear_annotations)
help_btn = tk.Button(control_frame, text='Help!', command=help_btn)
legend_btn = tk.Button(control_frame, text="Hide Legend", command=toggle_legend)
legend_btn.grid(row=1, column=0, columnspan=3, sticky='ew')
focus_btn = tk.Button(control_frame, text="Hide UI", command=toggle_focus_mode)
clear_all_btn = tk.Button(control_frame, text="Clear All", command=clear_all_datasets)


# Then, each button in its own column:
plot_btn.grid(row=0, column=0, padx=5, sticky='ew')
clear_btn.grid(row=0, column=1, padx=5, sticky='ew')
help_btn.grid(row=0, column=2, padx=5, sticky='ew')
legend_btn.grid(row=1, column=0, columnspan=3, sticky='ew')
focus_btn.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(5, 0))
clear_all_btn.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(5, 0))


root.mainloop()
