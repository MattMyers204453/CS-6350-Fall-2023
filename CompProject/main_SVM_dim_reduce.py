import os
import sys
import pandas as p
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer

categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]

train_file_path = os.path.join(sys.path[0], 'train_final.csv')
test_file_path = os.path.join(sys.path[0], 'test_final.csv')

print("reading dataset...")
df = p.read_csv(train_file_path, na_values="?")#.sample(n=1000, random_state=42)
df = df.dropna()
df_test = p.read_csv(test_file_path).drop('ID', axis=1)

print("converting categorical attributes to numerical...")
categorical_attributes = ["workclass","education","marital.status","occupation", "relationship","race", "sex","native.country"]  
column_trans = ColumnTransformer([('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_attributes)], remainder='passthrough')

X_train = column_trans.fit_transform(df.drop('income>50K', axis=1)) 
y_train = df['income>50K'].values
X_test = column_trans.transform(df_test)

print("standardizing and applying SVD...")
# pca = PCA(n_components=6)
#svd = TruncatedSVD(n_components=10)

scaler = StandardScaler(with_mean=False)
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Training...")
svm_model = SVC()
svm_model.fit(X_train, y_train)

print("testing...")
predictions = svm_model.predict(X_test).tolist()

ID_s = [1] * len(predictions)
for i in range(len(predictions)):
    ID_s[i] = i + 1
submission = p.DataFrame({'Id': ID_s, 'Prediction': predictions})
submission.to_csv("submission.csv", index=False)






