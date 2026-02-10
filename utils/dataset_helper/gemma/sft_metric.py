import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    pred_values = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)

    # Compute correlations
    pearson_corr, _ = pearsonr(labels, pred_values)
    spearman_corr, _ = spearmanr(labels, pred_values)
    kendall_corr, _ = kendalltau(labels, pred_values)

    # Mean of the three correlations
    mean_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "kendall": kendall_corr,
        "mean_corr": mean_corr
    }

if __name__ == "__main__":
    print(np.array(["1","2"],dtype=float))