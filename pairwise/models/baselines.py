"""
    Baseline machine learning models: random forest, extreme gradient boosting, extremely randomized tree, logistic regression
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

class baseline():
    def RF(self):
        model = RandomForestClassifier()
        return model

    def XGBOOST(self):
        model = GradientBoostingClassifier(max_features='auto')
        return model

    def ERT(self):
        model = ExtraTreesClassifier()
        return model

    def LR(self):
        model = LogisticRegression(solver='lbfgs')
        return model




        
