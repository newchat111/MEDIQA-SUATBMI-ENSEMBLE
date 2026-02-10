import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    id_to_label={'0': 0.0, '1': 0.5, '2': 1.0}

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()

    # Compute correlations
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    kendall_corr, _ = kendalltau(labels, predictions)

    # Mean of the three correlations
    mean_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])
    f1 = f1_score(labels, predictions, average='macro')

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "kendall": kendall_corr,
        "f1": f1,
        "mean_corr": mean_corr
    }