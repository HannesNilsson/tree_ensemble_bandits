import json
import xgboost as xgb

def get_leafs(tree: dict):
    """Get all leafs of a tree.

    Parameters
    ----------
    tree : dict
        Tree.

    Returns
    -------
    leafs : dict
        Dist of leafs.
    """

    leafs = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        try:
            stack.append(node['children'][0])
            stack.append(node['children'][1])
        except:
            leafs[node['nodeid']] = node #if node['nodeid'] != 0 else return leafs  

    return leafs

class XGBModel(xgb.XGBRegressor):
    def __init__(self, objective,
                 random_state, max_depth,
                 n_estimators, gamma, 
                 learning_rate, reg_lambda):
        
        super(XGBModel, self).__init__(objective=objective,
                 random_state=random_state, max_depth=max_depth,
                 n_estimators=n_estimators, gamma=gamma, 
                 learning_rate=learning_rate, reg_lambda=reg_lambda)