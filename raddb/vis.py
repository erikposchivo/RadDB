import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(embedding, labels=None, title="UMAP Projection", save_path=None):
    """
    Plots the UMAP embedding, optionally colored by labels.
    """
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Plot with classification colors
        sns.scatterplot(
            x=embedding[:, 0], 
            y=embedding[:, 1], 
            hue=labels, 
            palette="viridis", 
            s=5, 
            alpha=0.6,
            edgecolor=None
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Simple density plot
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5, c='steelblue', edgecolor=None)
    
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()
