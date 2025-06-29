# Placeholder for logic/selector.py
# logic/selector.py

def select_best_model(results):
    """
    Select the best model based on validation accuracy.

    Parameters:
        results (list): List of dictionaries from train_model()

    Returns:
        dict: Best model info
    """
    best_result = max(results, key=lambda x: x["val_accuracy"])
    return best_result
