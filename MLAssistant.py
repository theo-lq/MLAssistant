from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, VotingRegressor



class Assistant:
    
    
    def __init__(self, X, y, target_metric, metrics=None, cv=5):
        self.X = X
        self.y = y
        self.target_metric = target_metric
        self.metrics = [target_metric] if metrics == None else metrics
        self.cv=cv
        self.models = []
    
    
    
    def tryout(self, model_class, **kwargs):
        model = model_class(**kwargs)
        self._unit_performance(model)
    
    

    def _grid_search(self, model, parameters:dict):
        grid_search_model = GridSearchCV(model(), parameters, scoring=make_scorer(self.target_metric)).fit(self.X, self.y)
        best_estimator = grid_search_model.best_estimator_
        self._unit_performance(best_estimator)
        return best_estimator
    
    
    
    def _add_model(self, model):
        model_name = model.__class__.__name__
        identifier = model_name
        index = 1
        current_ids = [entry["id"] for entry in self.models]
        
        while identifier in current_ids:
            identifier = model_name + '_' + str(index)
            index += 1
        
        self.models.append({'id': identifier, 'model': model})
    
    

    def learn(self, models_parameters:list):
        best_models = [self._grid_search(model, parameters) for (model, parameters) in models_parameters]
        
        for model in best_models:
            self._add_model(model)
        
    
    
    
    def _unit_performance(self, model, show_name=True):
        if show_name:
            print(model.__class__.__name__)
            
        for metric in self.metrics:
            scores = cross_val_score(model, self.X, self.y, scoring=make_scorer(metric), cv=self.cv, n_jobs=-1)
            name = metric.__name__.replace("_score", "")
            print("%s: %0.2f (+/-%0.2f)" % (name, scores.mean(), scores.std() * 2), end="\t ")
        print(); print('-' * 110)
    
    
    
    def performance_recap(self):
        assert len(self.models) > 0, "No model is yet learned."
        
        for entry in self.models:
            print("id: %s - %s" % (entry["id"], entry["model"]))
            self._unit_performance(model=entry["model"], show_name=False)
    
    
    
    def delete_model(self, ids:list):
        old_list = [element for element in self.models]
        self.models = []
        for element in old_list:
            if not(element["id"] in ids):
                self.models.append(element)
    
    
    
    def make_ensemble(self):
        voting = VotingRegressor if len(self.y.value_counts()) > 100 else VotingClassifier
        estimators = [(entry["id"], entry["model"]) for entry in self.models]
        model = voting(estimators).fit(self.X, self.y)
        self._add_model(model)
    
    
    
    def predict(self, model_id:str, X_test, predict_proba=False):
        
        try:
            
            for element in self.models:
                if element["id"] == model_id:
                    model = element["model"]
            
            model.fit(self.X, self.y)
            
            if predict_proba:
                y_pred = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
            
            
        except UnboundLocalError as error:
            error_message = "'%s' is not in the model id list" % model_id
            raise NameError(error_message)

        
        return y_pred
        


