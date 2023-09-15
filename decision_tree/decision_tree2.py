import numpy as np
import pandas as pd

class DecisionTree:
    
    def __init__(self, max_depth=None):
        # Initialize any hyperparameters or attributes you need
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y, depth=0):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete values where
                each row is a sample, and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
            depth (int): Current depth in the tree (used for stopping criteria)
        """
        # Stopping criteria: If one of the following conditions is met, create a leaf node
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))

        # Find the best split based on information gain or other criteria
        best_split = self.find_best_split(X, y)

        # If no good split is found, create a leaf node
        if best_split is None:
            return np.argmax(np.bincount(y))

        # Split the data
        X_left, y_left, X_right, y_right = best_split

        # Recursively build the left and right subtrees
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        # Return a node that represents the decision at this level
        return {'feature_index': best_split[0],
                'threshold': best_split[1],
                'left': left_subtree,
                'right': right_subtree}
    
    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample, and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        predictions = []
        for _, row in X.iterrows():
            node = self.tree
            while isinstance(node, dict):
                feature_index = node['feature_index']
                threshold = node['threshold']
                if row[feature_index] <= threshold:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node)
        return np.array(predictions)
    
    def get_rules(self, node=None, path=[]):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjunction of attribute
        values, and the consequent is the predicted label

        attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            return [(path, str(node))]

        feature_index = node['feature_index']
        threshold = node['threshold']

        left_path = path + [(X.columns[feature_index], f'<= {threshold}')]
        right_path = path + [(X.columns[feature_index], f'> {threshold}')]

        left_rules = self.get_rules(node['left'], left_path)
        right_rules = self.get_rules(node['right'], right_path)

        return left_rules + right_rules

    def find_best_split(self, X, y):
        # Implement a method to find the best split based on information gain or other criteria
        pass
