 [file name]: create_figures_publication.py
[file content begin]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
import matplotlib

# Set publication-quality settings
matplotlib.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

print("="*70)
print("CREATING PUBLICATION-READY FIGURES")
print("(PDF for digital, TIFF for print)")
print("="*70)

# Load data
files = glob.glob('full_spec_results/*.parquet')
if len(files) == 0:
    print("ERROR: No data files found in full_spec_results/")
    print("Run at least one batch first: python3 run_batch_integrated.py Controlled 0")
    exit(1)

all_dfs = []
for f in files:
    df = pd.read_parquet(f)
    all_dfs.append(df)
    print(f"Loaded {f}: {len(df):,} rows")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal data: {len(df):,} rows from {df['run_id'].nunique()} runs")

# Create output directories
paper_dir = Path('paper_figures')
paper_dir.mkdir(exist_ok=True)

# Subdirectories for different formats
pdf_dir = paper_dir / 'pdf'
tiff_dir = paper_dir / 'tiff'
png_dir = paper_dir / 'png'  # Keep PNG for quick viewing
svg_dir = paper_dir / 'svg'  # SVG for editing

for dir_path in [pdf_dir, tiff_dir, png_dir, svg_dir]:
    dir_path.mkdir(exist_ok=True)

def save_all_formats(fig, base_name, dpi=300, fig_width_cm=None):
    """Save figure in all required formats with proper settings."""
    if fig_width_cm:
        # Convert cm to inches (1 inch = 2.54 cm)
        width_inches = fig_width_cm / 2.54
        fig.set_size_inches(width_inches, width_inches * 0.75)  # 4:3 aspect ratio
    
    # Save in all formats
    formats = {
        'pdf': {'format': 'pdf', 'dpi': dpi},
        'tiff': {'format': 'tiff', 'dpi': dpi, 
                'pil_kwargs': {'compression': 'tiff_lzw'}},
        'png': {'format': 'png', 'dpi': dpi},
        'svg': {'format': 'svg', 'dpi': dpi}
    }
    
    saved_files = []
    for fmt, params in formats.items():
        if fmt == 'tiff':
            # For TIFF, we need to use PIL backend
            fig.savefig(tiff_dir / f"{base_name}.{fmt}", **params)
        else:
            output_dir = locals()[f"{fmt}_dir"]  # Get the correct directory
            fig.savefig(output_dir / f"{base_name}.{fmt}", **params)
        saved_files.append(f"{fmt.upper()}: {base_name}.{fmt}")
    
    return saved_files

# Figure 1: Alert Rate Comparison (Single column: 8.6 cm width)
print("\n" + "="*70)
print("FIGURE 1: Alert Rate Comparison")
print("="*70)

plt.figure(figsize=(8.6/2.54, 6.5/2.54))  # Single column width
alert_by_run = df.groupby(['environment', 'run_id'])['alert_triggered'].mean().reset_index()

# Use colorblind-friendly palette
palette = {'Controlled': '#1f77b4', 'Residential': '#ff7f0e'}
sns.boxplot(x='environment', y='alert_triggered', data=alert_by_run, palette=palette)

plt.title('Alert Rate Distribution by Environment', pad=12)
plt.ylabel('Alert Rate')
plt.xlabel('Environment')

# Add sample size annotation
n_controlled = alert_by_run[alert_by_run['environment'] == 'Controlled'].shape[0]
n_residential = alert_by_run[alert_by_run['environment'] == 'Residential'].shape[0]
plt.text(0.5, 0.98, f'n={n_controlled}', transform=plt.gca().transAxes, 
         ha='center', va='top', fontsize=8)
plt.text(1.5, 0.98, f'n={n_residential}', transform=plt.gca().transAxes, 
         ha='center', va='top', fontsize=8)

# Save in all formats
saved_files = save_all_formats(plt.gcf(), 'figure1_alert_comparison', 
                               dpi=300, fig_width_cm=8.6)
for f in saved_files:
    print(f"  ✓ Saved {f}")

plt.close()

# Figure 2: Data Completeness (Single column: 8.6 cm width)
print("\n" + "="*70)
print("FIGURE 2: Data Completeness")
print("="*70)

plt.figure(figsize=(8.6/2.54, 6.5/2.54))
completeness = df.groupby('environment')['data_missing'].mean().reset_index()
completeness['completeness'] = (1 - completeness['data_missing']) * 100

# Calculate error bars (standard error)
error_bars = df.groupby('environment')['data_missing'].agg(['mean', 'sem']).reset_index()
error_bars['completeness'] = (1 - error_bars['mean']) * 100
error_bars['completeness_sem'] = error_bars['sem'] * 100

bars = sns.barplot(x='environment', y='completeness', data=completeness, 
                   palette=palette, edgecolor='black', linewidth=0.5)

# Add error bars manually
for i, env in enumerate(completeness['environment']):
    sem = error_bars[error_bars['environment'] == env]['completeness_sem'].values[0]
    plt.errorbar(i, completeness.loc[i, 'completeness'], 
                 yerr=sem, color='black', capsize=3, linewidth=1)

plt.title('Data Completeness by Environment', pad=12)
plt.ylabel('Data Completeness (%)')
plt.ylim(80, 100)
plt.xlabel('Environment')

# Add exact values on top of bars
for i, row in enumerate(completeness.itertuples()):
    plt.text(i, row.completeness + 0.5, f'{row.completeness:.1f}%', 
             ha='center', va='bottom', fontsize=8)

saved_files = save_all_formats(plt.gcf(), 'figure2_data_completeness', 
                               dpi=300, fig_width_cm=8.6)
for f in saved_files:
    print(f"  ✓ Saved {f}")

plt.close()

# Figure 3: Parameter Distributions (Double column: 17.8 cm width)
if df['hr_measured'].notna().any() and df['temp_measured'].notna().any():
    print("\n" + "="*70)
    print("FIGURE 3: Parameter Distributions (Double column)")
    print("="*70)
    
    plt.figure(figsize=(17.8/2.54, 7/2.54))  # Double column width
    
    # Heart Rate
    plt.subplot(1, 2, 1)
    for env in ['Controlled', 'Residential']:
        if env in df['environment'].unique():
            env_data = df[df['environment'] == env]['hr_measured'].dropna()
            if len(env_data) > 0:
                sns.histplot(env_data, bins=50, alpha=0.5, label=env, 
                           color=palette[env], stat='density', kde=True)
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Density')
    plt.legend(title='Environment')
    plt.grid(True, alpha=0.3)
    
    # Add statistics annotation
    hr_stats = df.groupby('environment')['hr_measured'].agg(['mean', 'std']).dropna()
    if not hr_stats.empty:
        stats_text = '\n'.join([f'{env}: {row["mean"]:.1f} ± {row["std"]:.1f} BPM' 
                               for env, row in hr_stats.iterrows()])
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Temperature
    plt.subplot(1, 2, 2)
    for env in ['Controlled', 'Residential']:
        if env in df['environment'].unique():
            env_data = df[df['environment'] == env]['temp_measured'].dropna()
            if len(env_data) > 0:
                sns.histplot(env_data, bins=50, alpha=0.5, label=env,
                           color=palette[env], stat='density', kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    plt.legend(title='Environment')
    plt.grid(True, alpha=0.3)
    
    # Add statistics annotation
    temp_stats = df.groupby('environment')['temp_measured'].agg(['mean', 'std']).dropna()
    if not temp_stats.empty:
        stats_text = '\n'.join([f'{env}: {row["mean"]:.1f} ± {row["std"]:.1f} °C' 
                               for env, row in temp_stats.iterrows()])
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    
    saved_files = save_all_formats(plt.gcf(), 'figure3_distributions', 
                                   dpi=300, fig_width_cm=17.8)
    for f in saved_files:
        print(f"  ✓ Saved {f}")
    
    plt.close()

# Create a README file with figure specifications
readme_content = f"""# Publication Figures
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data source: {len(files)} parquet files, {len(df):,} rows, {df['run_id'].nunique()} runs

## Figure Specifications:

### Figure 1: Alert Rate Comparison
- Dimensions: 8.6 cm × 6.5 cm (single column)
- DPI: 300
- Formats: PDF, TIFF, PNG, SVG
- Description: Box plot comparing alert rates between Controlled and Residential environments

### Figure 2: Data Completeness
- Dimensions: 8.6 cm × 6.5 cm (single column)
- DPI: 300
- Formats: PDF, TIFF, PNG, SVG
- Description: Bar chart showing data completeness percentage with error bars

### Figure 3: Parameter Distributions
- Dimensions: 17.8 cm × 7.0 cm (double column)
- DPI: 300
- Formats: PDF, TIFF, PNG, SVG
- Description: Histograms of heart rate and temperature distributions

## Color Palette:
- Controlled: Blue (#1f77b4)
- Residential: Orange (#ff7f0e)

## File Statistics:
- Total runs: {df['run_id'].nunique()}
- Environments: {sorted(df['environment'].unique().tolist())}
- Alert rate overall: {df['alert_triggered'].mean()*100:.1f}%
- Data completeness overall: {(1 - df['data_missing'].mean())*100:.1f}%

## Journal Submission Guidelines:
- Submit PDF files for digital/online version
- Submit TIFF files for print version
- PNG files are for quick viewing
- SVG files are for editing/modification

## Notes:
- All figures use colorblind-friendly palette
- Error bars represent standard error of the mean
- Statistical annotations included where applicable
"""

with open(paper_dir / 'README.md', 'w') as f:
    f.write(readme_content)

print("\n" + "="*70)
print("FIGURES CREATED SUCCESSFULLY")
print("="*70)
print("\n📁 Directory structure created:")
print("paper_figures/")
print("├── pdf/          # Publication-ready PDFs (digital)")
print("├── tiff/         # High-res TIFFs (print, 300 DPI)")
print("├── png/          # Quick view PNGs")
print("├── svg/          # Editable SVGs")
print("└── README.md     # Specifications and metadata")

print("\n📊 Figure statistics:")
for env in sorted(df['environment'].unique()):
    env_df = df[df['environment'] == env]
    alert_rate = env_df['alert_triggered'].mean() * 100
    completeness = (1 - env_df['data_missing'].mean()) * 100
    print(f"  {env}: Alert rate = {alert_rate:.1f}%, Completeness = {completeness:.1f}%")

print(f"\n✅ All figures saved with journal-ready specifications")
print(f"💡 For submission, upload PDFs for digital and TIFFs for print versions")
print("="*70)
[file content end]
