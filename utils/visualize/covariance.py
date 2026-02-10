import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_covariance(cov_df: pd.DataFrame, title: str = "Correlation Matrix", filename: str = "covariance_plot.png"):
    """
    Visualizes a single covariance/correlation matrix and saves it as an image.
    
    Args:
        cov_df (pd.DataFrame): The matrix to visualize.
        title (str): The title for the plot.
        filename (str): The name of the file to save.
    """
    # Initialize the figure
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(
        cov_df, 
        annot=True,          # Overlay the numeric values
        fmt=".2f",           # Format to 2 decimal places
        cmap='coolwarm',     # Red for positive, Blue for negative
        vmin=-1,             # Minimum value for color scale
        vmax=1,              # Maximum value for color scale
        square=True,         # Force square cells
        linewidths=.5,       # Add lines between cells for clarity
        cbar_kws={"shrink": .8}
    )
    
    # Styling
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save and close
    plt.savefig(filename)
    plt.close()
    
    return filename