import numpy as np
def compute_metrics(y_true, y_pred, y_bar, metrics, epsilon=1e-8):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    results = {}
    for metric in metrics:
        if metric.upper() == 'RMSP': # mean squared pearson residual
            result = np.sqrt(np.mean((y_true-y_pred)**2/(y_pred+epsilon)))
        elif metric.upper() == 'RMD': # mean squared deviance residual
            result = np.sqrt(np.mean(y_true*np.log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred)))
        elif metric.upper() == 'R2':
            numerator = np.mean(y_true*np.log((y_true+epsilon)/(y_pred+epsilon))-(y_true-y_pred))
            denominator = np.mean(y_true*np.log((y_true+epsilon)/(y_bar+epsilon))-(y_true-y_bar))
            result = 1 - numerator / denominator
        results[metric] = result
    return results