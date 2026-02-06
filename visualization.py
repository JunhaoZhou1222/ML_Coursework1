import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def get_data_states():
    print("Loading raw data...")
    # Before
    df_raw = pd.read_csv('CW1_train.csv')
    
    # After
    df_clean = df_raw.copy()
    
    # Carat Cleaning
    carat_outliers = df_clean[(df_clean['carat'] <= 0) | (df_clean['carat'] > 5)].index
    df_clean.drop(carat_outliers, inplace=True)
    
    # Depth Cleaning (IQR)
    depth_Q1 = df_clean["depth"].quantile(0.25)
    depth_Q3 = df_clean["depth"].quantile(0.75)
    depth_IQR = depth_Q3 - depth_Q1
    depth_lower = depth_Q1 - 1.5 * depth_IQR
    depth_upper = depth_Q3 + 1.5 * depth_IQR
    depth_outliers = df_clean[(df_clean['depth'] < depth_lower) | (df_clean['depth'] > depth_upper)].index
    df_clean.drop(depth_outliers, inplace=True)

    # Table Cleaning (IQR)
    table_Q1 = df_clean["table"].quantile(0.25)
    table_Q3 = df_clean["table"].quantile(0.75)
    table_IQR = table_Q3 - table_Q1
    table_lower = table_Q1 - 1.5 * table_IQR
    table_upper = table_Q3 + 1.5 * table_IQR
    table_outliers = df_clean[(df_clean['table'] < table_lower) | (df_clean['table'] > table_upper)].index
    df_clean.drop(table_outliers, inplace=True)

    # Dimensions Cleaning
    xyz_outliers = df_clean[(df_clean['x'] <= 0) | (df_clean['y'] <= 0) | (df_clean['z'] <= 0)].index
    df_clean.drop(xyz_outliers, inplace=True)
    
    return df_raw, df_clean

def plot_comparisons(df_raw, df_clean):
    print("Generating visualizations...")
    
    # Boxplots to show IQR for Depth and Table
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Depth Comparison
    sns.boxplot(x=df_raw['depth'], ax=axes[0, 0], color='salmon')
    axes[0, 0].set_title('Depth: Before Cleaning (Original)')
    
    sns.boxplot(x=df_clean['depth'], ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Depth: After IQR Cleaning')
    
    # Table Comparison
    sns.boxplot(x=df_raw['table'], ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title('Table: Before Cleaning (Original)')
    
    sns.boxplot(x=df_clean['table'], ax=axes[1, 1], color='lightgreen')
    axes[1, 1].set_title('Table: After IQR Cleaning')
    
    plt.tight_layout()
    plt.savefig('viz_comparison_boxplots.png')
    print("Saved viz_comparison_boxplots.png")
    
    # Scatter Plot show Depth vs Table before and after cleaning
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before
    axes[0].scatter(df_raw['table'], df_raw['depth'], alpha=0.3, s=10, c='red', label='Outliers')
    axes[0].set_title(f'Depth vs Table (Raw Data, N={len(df_raw)})')
    axes[0].set_xlabel('Table')
    axes[0].set_ylabel('Depth')
    
    # After
    axes[1].scatter(df_clean['table'], df_clean['depth'], alpha=0.3, s=10, c='green', label='Cleaned')
    axes[1].set_title(f'Depth vs Table (Cleaned Data, N={len(df_clean)})')
    axes[1].set_xlabel('Table')
    axes[1].set_ylabel('Depth')
    
    # Set same limits for better comparison
    xlims = (min(df_raw['table'].min(), df_clean['table'].min()), max(df_raw['table'].max(), df_clean['table'].max()))
    ylims = (min(df_raw['depth'].min(), df_clean['depth'].min()), max(df_raw['depth'].max(), df_clean['depth'].max()))
    axes[0].set_xlim(xlims); axes[0].set_ylim(ylims)
    axes[1].set_xlim(xlims); axes[1].set_ylim(ylims)

    plt.tight_layout()
    plt.savefig('viz_comparison_scatter.png')
    print("Saved viz_comparison_scatter.png")

    #show Carat distribution before and after cleaning
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_raw['carat'], fill=True, color='red', label='Raw Data')
    sns.kdeplot(df_clean['carat'], fill=True, color='green', label='Cleaned Data')
    plt.title('Carat Distribution Change')
    plt.legend()
    plt.savefig('viz_comparison_carat.png')
    print("Saved viz_comparison_carat.png")

if __name__ == "__main__":
    raw_df, clean_df = get_data_states()
    
    print(f"Original Rows: {len(raw_df)}")
    print(f"Cleaned Rows:  {len(clean_df)}")
    print(f"Dropped Rows:  {len(raw_df) - len(clean_df)}")
    
    plot_comparisons(raw_df, clean_df)