"""
analyze_attention_results_fixed.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set default font for better compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

PRODUCTS = ['ERA5', 'CoLM', 'SMCI']
NUM_HEADS = 4

class AttentionAnalyzer:
    def __init__(self, data_dir='./analyze_attention'):
        self.data_dir = data_dir
        self.load_data()
        self.load_mask()

    def load_data(self):
        """Load all data"""
        print("Loading data...")

        # Load full distributions
        with open(f'{self.data_dir}/attn_full_distributions.pkl', 'rb') as f:
            self.distributions = pickle.load(f)

        # Load statistics
        self.mean_attn = np.load(f'{self.data_dir}/attn_stats_mean.npy')  # (lat, lon, heads, 3, 3)
        self.std_attn = np.load(f'{self.data_dir}/attn_stats_std.npy')
        self.n_samples = np.load(f'{self.data_dir}/attn_stats_n_samples.npy')

        # Load errors
        self.error_era5 = np.load(f'{self.data_dir}/avg_error_era5_per_grid.npy')
        self.error_colm = np.load(f'{self.data_dir}/avg_error_colm_per_grid.npy')
        self.error_smci = np.load(f'{self.data_dir}/avg_error_smci_per_grid.npy')

        # Load metadata
        with open(f'{self.data_dir}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"Loaded: {len(self.distributions)} grid points")
        print(f"Attention shape: {self.mean_attn.shape}")
        print(f"NaN count in SMCI error: {np.sum(np.isnan(self.error_smci))}")

    def load_mask(self):
        """Load Northeast China mask"""
        mask_path = '/home/zhangcheng/Soil_Moisture/CML_FD/dataset/mask_Northeast_China.npy'
        self.china_mask = np.load(mask_path)
        print(f"Mask shape: {self.china_mask.shape}")
        print(f"Grid points within mask: {np.sum(self.china_mask == 1)}")

        # Get valid grid indices within mask
        self.valid_mask_indices = np.where(self.china_mask == 1)
        self.valid_grids_in_mask = list(zip(self.valid_mask_indices[0], self.valid_mask_indices[1]))

        # Filter distributions to keep only mask points
        self.filtered_distributions = {}
        for (i, j) in self.valid_grids_in_mask:
            if (i, j) in self.distributions:
                self.filtered_distributions[(i, j)] = self.distributions[(i, j)]

        print(f"Valid data grid points within mask: {len(self.filtered_distributions)}")

        # Create masked versions of statistics
        self.mean_attn_masked = self.mean_attn.copy().astype(np.float32)
        self.error_era5_masked = self.error_era5.copy().astype(np.float32)
        self.error_colm_masked = self.error_colm.copy().astype(np.float32)
        self.error_smci_masked = self.error_smci.copy().astype(np.float32)

        # Set outside-mask regions to NaN
        for i in range(self.mean_attn.shape[0]):
            for j in range(self.mean_attn.shape[1]):
                if self.china_mask[i, j] != 1:
                    self.mean_attn_masked[i, j] = np.nan
                    self.error_era5_masked[i, j] = np.nan
                    self.error_colm_masked[i, j] = np.nan
                    self.error_smci_masked[i, j] = np.nan

    def basic_statistics(self):
        """1. Basic statistics analysis (masked region only)"""
        print("\n" + "="*60)
        print("1. Basic Statistics Analysis (Northeast China)")
        print("="*60)

        # Collect attention weights within mask
        all_attn = []
        all_errors = []

        max_grids = min(1000, len(self.filtered_distributions))
        grid_list = list(self.filtered_distributions.keys())[:max_grids]

        for grid in grid_list:
            data = self.filtered_distributions[grid]
            attn_flat = data['attn'].reshape(-1, NUM_HEADS*9)
            all_attn.append(attn_flat)
            all_errors.append(data['errors'])

        # Average attention matrix per head
        print("\nAverage attention matrix per head (masked region):")
        for head in range(NUM_HEADS):
            masked_mean = self.mean_attn_masked[..., head, :, :]
            head_mean = np.nanmean(masked_mean, axis=(0,1))
            print(f"\nHead {head}:")
            df = pd.DataFrame(head_mean, index=PRODUCTS, columns=PRODUCTS)
            print(df.round(3))

        # Attention weight distribution statistics
        if len(all_attn) > 0:
            all_attn_array = np.concatenate(all_attn, axis=0)
            all_errors_array = np.concatenate(all_errors, axis=0)

            all_attn_array = all_attn_array.astype(np.float32)
            all_errors_array = all_errors_array.astype(np.float32)

            print("\nAttention weight distribution statistics (masked region):")
            attn_flat = all_attn_array.flatten()
            print(f"  Mean: {np.mean(attn_flat):.4f}")
            print(f"  Std: {np.std(attn_flat):.4f}")
            print(f"  Min: {np.min(attn_flat):.4f}")
            print(f"  Max: {np.max(attn_flat):.4f}")
            print(f"  Skewness: {stats.skew(attn_flat):.4f}")
            print(f"  Kurtosis: {stats.kurtosis(attn_flat):.4f}")

            # Error statistics
            print("\nPrediction error statistics per product (Day 1, masked region):")
            for i, prod in enumerate(PRODUCTS):
                errors = all_errors_array[:, i]
                if not np.all(np.isnan(errors)):
                    print(f"\n{prod}:")
                    print(f"  Mean: {np.nanmean(errors):.4f}")
                    print(f"  Std: {np.nanstd(errors):.4f}")
                    print(f"  Median: {np.nanmedian(errors):.4f}")
                    print(f"  95th percentile: {np.nanpercentile(errors, 95):.4f}")
                else:
                    print(f"\n{prod}: All errors are NaN")
        else:
            print("\nWarning: No valid attention data collected")

    def plot_spatial_patterns(self):
        """2. Spatial pattern visualization (masked region only)"""
        print("\n" + "=" * 60)
        print("2. Spatial Pattern Visualization (Northeast China)")
        print("=" * 60)

        # Calculate self-attention
        self_attn = np.zeros((self.mean_attn_masked.shape[0],
                              self.mean_attn_masked.shape[1], NUM_HEADS))
        for head in range(NUM_HEADS):
            for i in range(3):
                self_attn[..., head] += self.mean_attn_masked[..., head, i, i]
            self_attn[..., head] /= 3

       
        all_data = []
        for head in range(NUM_HEADS):
            data = self_attn[..., head]
            all_data.append(data[~np.isnan(data)])

        
        if len(np.concatenate(all_data)) > 0:
            global_vmin = np.percentile(np.concatenate(all_data), 2)
            global_vmax = np.percentile(np.concatenate(all_data), 98)
        else:
            global_vmin = 0.3
            global_vmax = 0.37

        print(f"统一颜色范围: vmin={global_vmin:.4f}, vmax={global_vmax:.4f}")

        # Self-attention spatial distribution with unified colorbar
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Self-Attention Weight Spatial Distribution (Northeast China)', fontsize=16)

        for head in range(NUM_HEADS):
            ax = axes[head // 2, head % 2]
            data = self_attn[..., head]

            
            im = ax.imshow(data, cmap='RdYlBu_r',
                           vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f'Head {head} - Mean Self-Attention')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            
            plt.colorbar(im, ax=ax, label='Attention Weight')

        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/spatial_self_attention_masked.png', dpi=150, bbox_inches='tight')
        plt.show()

        
        fig, ax = plt.subplots(figsize=(10, 6))

        
        head_data = []
        head_names = []
        for head in range(NUM_HEADS):
            data = self_attn[..., head]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                head_data.append(valid_data)
                head_names.append(f'Head {head}')

        if head_data:
            bp = ax.boxplot(head_data, labels=head_names, patch_artist=True, showmeans=True)

           
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(head_data)]):
                patch.set_facecolor(color)

            ax.set_ylabel('Self-Attention Weight')
            ax.set_title('Distribution of Self-Attention Weights by Head')
            ax.grid(True, alpha=0.3)

            ax.axhline(y=global_vmin, color='red', linestyle='--', alpha=0.5, label=f'Global vmin ({global_vmin:.3f})')
            ax.axhline(y=global_vmax, color='red', linestyle='--', alpha=0.5, label=f'Global vmax ({global_vmax:.3f})')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{self.data_dir}/self_attention_distribution_comparison.png', dpi=150, bbox_inches='tight')
            plt.show()





        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Prediction Error Spatial Distribution (Day 1, Northeast China)', fontsize=16)

        errors = [self.error_era5_masked, self.error_colm_masked, self.error_smci_masked]
        for idx, (ax, error, prod) in enumerate(zip(axes, errors, PRODUCTS)):
            if not np.all(np.isnan(error)):
                vmin = np.nanpercentile(error, 5)
                vmax = np.nanpercentile(error, 95)
                im = ax.imshow(error, cmap='YlOrRd', vmin=vmin, vmax=vmax)
                ax.set_title(f'{prod}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax, label='MAE')
            else:
                ax.set_title(f'{prod} (No Data)')
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center')

        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/spatial_errors_masked.png', dpi=150, bbox_inches='tight')
        plt.show()


        if not np.all(np.isnan(self.error_era5_masked)) and not np.all(np.isnan(self.error_colm_masked)):
            fig, ax = plt.subplots(figsize=(10, 8))
            diff_era5_colm = self.error_era5_masked - self.error_colm_masked
            diff_max = max(abs(np.nanpercentile(diff_era5_colm, 5)),
                           abs(np.nanpercentile(diff_era5_colm, 95)))
            im = ax.imshow(diff_era5_colm, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
            ax.set_title('ERA5 vs CoLM Error Difference (Positive = ERA5 Larger Error)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='MAE Difference')
            plt.savefig(f'{self.data_dir}/error_difference_masked.png', dpi=150, bbox_inches='tight')
            plt.show()

    def attention_error_correlation(self):
        """3. Attention-error correlation analysis (masked region only)"""
        print("\n" + "="*60)
        print("3. Attention-Error Correlation Analysis (Northeast China)")
        print("="*60)

        # Collect statistics per grid point
        grid_stats = []
        grid_list = list(self.filtered_distributions.keys())[:500]

        for (i, j) in grid_list:
            data = self.filtered_distributions[(i, j)]

            mean_attn = np.mean(data['attn'], axis=0).astype(np.float32)
            mean_error = np.mean(data['errors'], axis=0).astype(np.float32)

            if np.any(np.isnan(mean_error)):
                continue

            grid_stats.append({
                'lat': i, 'lon': j,
                'mean_error_era5': mean_error[0],
                'mean_error_colm': mean_error[1],
                'mean_error_smci': mean_error[2],
                'self_attn_mean': np.mean([mean_attn[h, k, k] for h in range(NUM_HEADS) for k in range(3)]),
                'cross_attn_mean': np.mean([mean_attn[h, i, j] for h in range(NUM_HEADS)
                                          for i in range(3) for j in range(3) if i != j]),
                'n_samples': data['n_samples']
            })

        if len(grid_stats) > 0:
            grid_stats_df = pd.DataFrame(grid_stats)

            print("\nSelf-attention vs prediction error correlation:")
            for prod in ['era5', 'colm', 'smci']:
                error_col = f'mean_error_{prod}'
                if error_col in grid_stats_df.columns:
                    valid_data = grid_stats_df[['self_attn_mean', error_col]].dropna()
                    if len(valid_data) > 0:
                        corr = valid_data['self_attn_mean'].corr(valid_data[error_col])
                        print(f"  {prod.upper()}: r = {corr:.4f} (n={len(valid_data)})")
                    else:
                        print(f"  {prod.upper()}: Insufficient valid data")

            print("\nCross-attention vs prediction error correlation:")
            for prod in ['era5', 'colm', 'smci']:
                error_col = f'mean_error_{prod}'
                if error_col in grid_stats_df.columns:
                    valid_data = grid_stats_df[['cross_attn_mean', error_col]].dropna()
                    if len(valid_data) > 0:
                        corr = valid_data['cross_attn_mean'].corr(valid_data[error_col])
                        print(f"  {prod.upper()}: r = {corr:.4f} (n={len(valid_data)})")
                    else:
                        print(f"  {prod.upper()}: Insufficient valid data")

            # Scatter plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Self-Attention Weight vs Prediction Error (Northeast China)', fontsize=16)

            for idx, prod in enumerate(PRODUCTS):
                ax = axes[idx]
                error_col = f'mean_error_{prod.lower()}'

                if error_col in grid_stats_df.columns:
                    valid_data = grid_stats_df[['self_attn_mean', error_col]].dropna()

                    if len(valid_data) > 0:
                        ax.scatter(valid_data['self_attn_mean'], valid_data[error_col],
                                  alpha=0.5, s=20, c='blue')

                        if len(valid_data) > 10:
                            z = np.polyfit(valid_data['self_attn_mean'], valid_data[error_col], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(valid_data['self_attn_mean'].min(),
                                                 valid_data['self_attn_mean'].max(), 100)
                            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8,
                                   label=f'Trend line (r={valid_data["self_attn_mean"].corr(valid_data[error_col]):.3f})')

                        ax.set_xlabel('Mean Self-Attention Weight')
                        ax.set_ylabel('MAE')
                        ax.set_title(f'{prod} (n={len(valid_data)})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, f'No valid data for {prod}',
                               transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, f'No data for {prod}',
                           transform=ax.transAxes, ha='center', va='center')

            plt.tight_layout()
            plt.savefig(f'{self.data_dir}/attention_vs_error_scatter_masked.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            print("Insufficient grid points for correlation analysis")

    def compare_attention_patterns(self):
        """4. Product attention pattern comparison (masked region only)"""
        print("\n" + "="*60)
        print("4. Product Attention Pattern Comparison (Northeast China)")
        print("="*60)

        # Calculate average attention matrix within mask
        mean_overall = np.nanmean(self.mean_attn_masked, axis=(0,1))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Average Attention Pattern per Head (Northeast China)', fontsize=16)

        for head in range(NUM_HEADS):
            ax = axes[head//2, head%2]
            im = ax.imshow(mean_overall[head], cmap='Blues', vmin=0, vmax=0.6)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(PRODUCTS)
            ax.set_yticklabels(PRODUCTS)
            ax.set_title(f'Head {head}')

            # Add value labels
            for i in range(3):
                for j in range(3):
                    text = ax.text(j, i, f'{mean_overall[head, i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10)

            plt.colorbar(im, ax=ax, label='Attention Weight')

        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/attention_heatmaps_masked.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Average attention when each product serves as query
        print("\nAverage attention distribution per query product (masked region):")
        for prod_idx, prod in enumerate(PRODUCTS):
            query_attn = np.mean([mean_overall[head, prod_idx, :] for head in range(NUM_HEADS)], axis=0)
            print(f"\nQuery: {prod}")
            for target_idx, target in enumerate(PRODUCTS):
                print(f"  → {target}: {query_attn[target_idx]:.4f}")

    def analyze_regional_variation(self):
        """5. Regional variation analysis"""
        print("\n" + "="*60)
        print("5. Regional Variation Analysis (Northeast China)")
        print("="*60)

        # Calculate self-attention for each grid point
        self_attn_all = np.zeros((self.mean_attn_masked.shape[0],
                                  self.mean_attn_masked.shape[1], NUM_HEADS))
        for head in range(NUM_HEADS):
            for i in range(3):
                self_attn_all[..., head] += self.mean_attn_masked[..., head, i, i]
            self_attn_all[..., head] /= 3

        self_attn_mean = np.nanmean(self_attn_all, axis=2)

        # Calculate coefficient of variation
        cv = np.nanstd(self_attn_all, axis=2) / (np.nanmean(self_attn_all, axis=2) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Attention Pattern Regional Variation (Northeast China)', fontsize=16)

        # Mean self-attention
        im1 = axes[0].imshow(self_attn_mean, cmap='RdYlBu_r',
                            vmin=np.nanpercentile(self_attn_mean, 5),
                            vmax=np.nanpercentile(self_attn_mean, 95))
        axes[0].set_title('Mean Self-Attention Weight')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')

        # Coefficient of variation
        im2 = axes[1].imshow(cv, cmap='YlOrRd',
                            vmin=np.nanpercentile(cv, 5),
                            vmax=np.nanpercentile(cv, 95))
        axes[1].set_title('Coefficient of Variation (CV)')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1], label='CV')

        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/regional_variation_masked.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nSelf-attention statistics:")
        print(f"  Mean: {np.nanmean(self_attn_mean):.4f}")
        print(f"  Std: {np.nanstd(self_attn_mean):.4f}")
        print(f"  CV: {np.nanmean(cv):.4f}")

    def generate_report(self):
        """Generate analysis report (masked region only)"""
        print("\n" + "="*60)
        print("Generating Analysis Report (Northeast China)")
        print("="*60)

        # Calculate key statistics
        mean_overall = np.nanmean(self.mean_attn_masked, axis=(0,1))

        report = []
        report.append("="*60)
        report.append("Attention Weight Analysis Report - Northeast China")
        report.append("="*60)
        report.append(f"\nData Overview:")
        report.append(f"  - Number of grid points analyzed: {len(self.filtered_distributions)}")
        report.append(f"  - Number of attention heads: {NUM_HEADS}")
        report.append(f"  - Region: Northeast China")

        report.append(f"\nAverage Attention Matrix per Head (Masked Region):")
        for head in range(NUM_HEADS):
            report.append(f"\nHead {head}:")
            for i in range(3):
                row = [f"{mean_overall[head, i, j]:.3f}" for j in range(3)]
                report.append(f"  {PRODUCTS[i]}: " + " ".join(row))

        # Find strongest attention connection
        if not np.all(np.isnan(mean_overall)):
            max_attn = np.unravel_index(np.nanargmax(mean_overall), mean_overall.shape)
            report.append(f"\nStrongest Attention Connection:")
            report.append(f"  Head {max_attn[0]}: {PRODUCTS[max_attn[1]]} → {PRODUCTS[max_attn[2]]} "
                         f"(weight={mean_overall[max_attn]:.4f})")

        # Error statistics
        report.append(f"\nPrediction Error Statistics (Day 1):")
        for prod, error in zip(PRODUCTS, [self.error_era5_masked, self.error_colm_masked, self.error_smci_masked]):
            if not np.all(np.isnan(error)):
                mean_err = np.nanmean(error)
                std_err = np.nanstd(error)
                report.append(f"  {prod}: {mean_err:.4f} ± {std_err:.4f}")
            else:
                report.append(f"  {prod}: No valid data")

        # Save report
        with open(f'{self.data_dir}/analysis_report_masked.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print("\n".join(report))

def main():
    analyzer = AttentionAnalyzer('./analyze_attention')

    # Run analyses (masked region only)
    analyzer.basic_statistics()
    analyzer.plot_spatial_patterns()
    analyzer.attention_error_correlation()
    analyzer.compare_attention_patterns()
    analyzer.analyze_regional_variation()
    analyzer.generate_report()

    print("\n" + "="*60)
    print("Analysis complete! Results saved (Northeast China region only)")
    print("="*60)

if __name__ == "__main__":
    main()
