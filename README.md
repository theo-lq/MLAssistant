# MLAssistant
The ML Assistant class's aim is to speed and simplify the models training and comparison during test phase of algorithm. The assistant store algorithm that have been learned so that one can always keep an eye on them. One can also remove these algorithm in order to pick different models and make a voting algorithm out of it easily.

The main pipeline of use of the ML Assistant, once the dataset is ready to be learned (i.e after preprocessing and feature engineering), is as follow :

1. **Tryout** of models: use the *tryout* method to check the performance one can have with a given algorithm with given parameters
2. **Learn** some models: use the *learn* method to find the best parameters for the algorithm the data scientist want to train, and then store the model learned
3. **Compare** performance of models stored: use the *performance_recap* method to check the performance of each algorithm stored
4. **Remove** models stored: use the *delete_model* method to remove the models that the data scientist choosed
5. Create a **ensemble** model: use the *make_ensemble* method to create a Voting Classifier/Regressor with the models learned and currently stored
6. **Predict**: use the *predict* method with the 'id' of the model the data scientist choosed

One can see an example with the notebook *Use of the Assistant class* in the repository.
