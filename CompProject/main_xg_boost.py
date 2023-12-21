import os
import sys
import pandas as p
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
import xgboost

def replace_surprise_values(column):
    return column.where(column.isin(X_train[column.name]), modes[column.name])

categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]

train_file_path = os.path.join(sys.path[0], 'train_final.csv')
test_file_path = os.path.join(sys.path[0], 'test_final.csv')

print("reading dataset...")
df = p.read_csv(train_file_path, na_values="?")#.sample(n=1000, random_state=42)
#train_df = train_df.dropna()
df_test = p.read_csv(test_file_path).drop('ID', axis=1)

X_train = df.drop('income>50K', axis=1)
y_train = df['income>50K'].values
X_test = df_test

modes = X_train.mode().iloc[0]
X_test = X_test.apply(replace_surprise_values)

# Encode each value for the categorical data into an integer
encoder = OrdinalEncoder()
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print("Training...")
xgb_model = xgboost.XGBClassifier(eval_metric='logloss')

hyperparameter_grid = {
    'n_estimators': [100, 500, 700],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=hyperparameter_grid, cv=4, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
optimal_hyper_parameters = grid_search.best_params_

xgb_model_optimal = xgboost.XGBClassifier(**optimal_hyper_parameters, eval_metric='logloss')
xgb_model_optimal.fit(X_train, y_train)

print("testing...")
predictions = xgb_model_optimal.predict(X_test).tolist()

ID_s = [1] * len(predictions)
for i in range(len(predictions)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)





