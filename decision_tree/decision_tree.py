import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__():
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, depth = 0):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement 
        
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        
        best_split, best_gini = None, 1.0
        for feature_index in range(n_features):
            unique_vals = np.unique(X[:,feature_index])
            for val in unique_vals:
                left_index = np.where(X[:, feature_index] == val)[0]
                right_index = np.where(X[:,feature_index] != val)[0]
                if len(left_index) == 0 or len(right_index) == 0:
                    continue
                p_l = len(y[left_index]) / (len(y[left_index]) + len(y[right_index]))
                p_r = len(y[right_index]) / (len(y[right_index]) + len(y[left_index]))
                gini_l = 1.0 - sum((np.sum(y[left_index] == c) / len(y[left_index]))**2 for c in np.unique(y[left_index]))
                gini_r = 1.0 - sum((np.sum(y[right_index] == c) / len(y[right_index]))**2 for c in np.unique(y[right_index]))
                gini = p_l * gini_l + p_r * gini_r
                if gini < best_gini:
                    best_split, best_gini = (feature_index, val), gini
        if best_gini == 1.0:
            return unique_classes[np.argmax(np.bincount(y))]
        left_index, right_index = np.where(X[:,best_split[0]] == best_split[1])[0], np.where(X[:,best_split[0]] != best_split[1])[0]
        left_subtree = self.fit(X[left_index], y[left_index], depth + 1)
        right_subtree = self.fit(X[right_index], y[right_index], depth + 1)
        
        return (best_split, left_subtree, right_subtree)
        
        #raise NotImplementedError()
    
    def predict(self, X: pd.DataFrame):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        
        
        
        raise NotImplementedError()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



