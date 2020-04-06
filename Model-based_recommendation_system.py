import pandas as pd

import math

import time
import numpy as np
import xgboost as xgb
from sklearn import model_selection, preprocessing
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from pyspark import SparkConf, SparkContext
import sys
import json
from collections import OrderedDict


start = time.time()

training_folder_path=sys.argv[1]
testing_file_path=sys.argv[2]
output_file_path=sys.argv[3]


user_file=pd.read_json(training_folder_path+"/user.json",lines=True,chunksize=100000)
for user_chunk in user_file:
    first_user_data_chunk=user_chunk
    break

all_user_data=first_user_data_chunk.copy()

flag=0
for user_chunk in user_file:
    all_user_data=all_user_data.append(user_chunk)


# print(all_user_data.shape)
# print(all_user_data.head())


business_file=pd.read_json(training_folder_path+"/business.json",lines=True,chunksize=100000)
for business_chunk in business_file:
    first_business_data_chunk=business_chunk
    break

all_business_data=first_business_data_chunk.copy()

flag=0
for business_chunk in business_file:
    all_business_data=all_business_data.append(business_chunk)



# tip_file=pd.read_json(training_folder_path+"/tip.json",lines=True,chunksize=100000)
# for tip_chunk in tip_file:
#     first_tip_data_chunk=tip_chunk
#     break
#
# all_tip_data=first_tip_data_chunk.copy()
#
# flag=0
# for tip_chunk in tip_file:
#     all_tip_data=all_tip_data.append(tip_chunk)


#
# print(first_user_data_chunk.shape)
# print(first_user_data_chunk.dtypes)
# print(first_user_data_chunk.head(5))

train_set = pd.read_csv(training_folder_path+"/yelp_train.csv")
# print(train_set.head())

#,'funny','cool','compliment_hot','compliment_more','compliment_profile','compliment_cute','compliment_list','compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos'
data_required_from_user=all_user_data[['user_id','average_stars','review_count','useful','fans']]
# print(data_required_from_user.dtypes)
data_required_from_business=all_business_data[['business_id','stars','review_count']]
# data_required_from_business['postal_code']=pd.to_numeric(data_required_from_business['postal_code'])
data_required_from_business=data_required_from_business.rename(columns={'stars':'business_avg_stars','review_count':'business_rev_count'})
# print(data_required_from_business.dtypes)

# print(data_required_from_business.head())


# data_required_from_tip=all_tip_data[['user_id','business_id','likes']]
# print(data_required_from_tip.dtypes)
# print(data_required_from_tip.head())



join_of_requiredUserData_and_train=pd.merge(train_set,data_required_from_user,on='user_id',how='inner')
join_of_requiredUserData_and_train=pd.merge(join_of_requiredUserData_and_train,data_required_from_business,on='business_id',how='inner')

totalOfSquaredStars_user_ratings_dictionary={}
user_to_totalStars_dictionary={}
user_to_reviewCount_dictionary={}
user_to_variance_dictionary={}
user_to_userAvgStars_dictionary={}
var_numerator_dictionary={}
user_to_mode_dictionary={}
user_to_max_dictionary={}
user_to_min_dictionary={}

for row_index, row in join_of_requiredUserData_and_train.iterrows():
    if(row['user_id'] not in totalOfSquaredStars_user_ratings_dictionary):
        # totalOfSquaredStars_user_ratings_dictionary[row['user_id']]=math.pow(row['stars'],2)
        # user_to_totalStars_dictionary[row['user_id']]=row['stars']
        user_to_reviewCount_dictionary[row['user_id']]=1
        # user_to_userAvgStars_dictionary[row['user_id']]=row['average_stars']
        var_numerator_dictionary[row['user_id']]=math.pow(row['stars']-row['average_stars'],2)
        user_to_max_dictionary[row['user_id']]=row['stars']
        user_to_min_dictionary[row['user_id']] = row['stars']

    else:
        # totalOfSquaredStars_user_ratings_dictionary[row['user_id']]+=math.pow(row['stars'],2)
        # user_to_totalStars_dictionary[row['user_id']]+=row['stars']
        user_to_reviewCount_dictionary[row['user_id']]+=1
        var_numerator_dictionary[row['user_id']]+= math.pow(row['stars'] - row['average_stars'], 2)
        if(row['stars']>user_to_max_dictionary[row['user_id']]):
            user_to_max_dictionary[row['user_id']]=row['stars']
        if (row['stars'] < user_to_min_dictionary[row['user_id']]):
            user_to_min_dictionary[row['user_id']] = row['stars']


user_stats_dataFrame=pd.DataFrame(columns=["user_id","user_variance","user_max","user_min"])

for current_user in user_to_reviewCount_dictionary:
    # mean = user_to_totalStars_dictionary[current_user] / user_to_reviewCount_dictionary[current_user]
    # mean=user_to_userAvgStars_dictionary[current_user]
    # current_variance = (totalOfSquaredStars_user_ratings_dictionary[current_user] / (user_to_reviewCount_dictionary[
    #     current_user]-1)) -(user_to_reviewCount_dictionary[current_user]/(user_to_reviewCount_dictionary[current_user]-1)*(math.pow(mean, 2)))
    current_variance=var_numerator_dictionary[current_user]/user_to_reviewCount_dictionary[current_user]
    user_stats_dataFrame=user_stats_dataFrame.append({'user_id': current_user,'user_variance':current_variance,'user_max':user_to_max_dictionary[current_user],'user_min':user_to_min_dictionary[current_user]},ignore_index=True)


print(user_stats_dataFrame.dtypes)
print(user_stats_dataFrame.head())

join_of_requiredUserData_and_train=pd.merge(join_of_requiredUserData_and_train,user_stats_dataFrame,on='user_id',how='left')









# join_of_requiredUserData_and_train=pd.merge(join_of_requiredUserData_and_train,data_required_from_tip,on=['user_id','business_id'],how='inner')
# print(join_of_requiredUserData_and_train[['user_id','business_id','review_count']].head(30))

processed_join_train=join_of_requiredUserData_and_train.copy()

for column_name in processed_join_train.columns:
    if processed_join_train[column_name].dtype=='object':
        # if column_name=="yelping_since":
        #     continue
        label_encoder=preprocessing.LabelEncoder()
        label_encoder.fit(list(processed_join_train[column_name].values))
        processed_join_train[column_name]=label_encoder.transform(list(processed_join_train[column_name].values))

# print(processed_join_train.tail())

training_data_result_ratings=processed_join_train.stars.values
training_data_input=processed_join_train.drop(['stars'],axis=1)
training_data_input=training_data_input.drop(['user_id'],axis=1)
training_data_input=training_data_input.drop(['business_id'],axis=1)
training_data_input=training_data_input.values


ratings_predictor_model=xgb.XGBRegressor()
ratings_predictor_model.fit(training_data_input,training_data_result_ratings)
# print(training_data_input[:10])
# print(training_data_result_ratings[:10])



# print(ratings_predictor_model)

test_set = pd.read_csv(testing_file_path)

join_of_requiredUserData_and_test=pd.merge(test_set,data_required_from_user,on='user_id',how='left')
join_of_requiredUserData_and_test=pd.merge(join_of_requiredUserData_and_test,data_required_from_business,on='business_id',how='left')
join_of_requiredUserData_and_test=pd.merge(join_of_requiredUserData_and_test,user_stats_dataFrame,on='user_id',how='left')

# join_of_requiredUserData_and_test=pd.merge(join_of_requiredUserData_and_test,data_required_from_tip,on=['user_id','business_id'],how='left')

# print(join_of_requiredUserData_and_test[['user_id','business_id','review_count']].head(30))

user_key_values_of_our_data=join_of_requiredUserData_and_test['user_id'].values
business_key_values_of_our_data=join_of_requiredUserData_and_test['business_id'].values

# print(user_key_values_of_our_data[:10])
# print(business_key_values_of_our_data[:10])



processed_join_test=join_of_requiredUserData_and_test.copy()

for column_name in processed_join_test.columns:
    if processed_join_test[column_name].dtype=='object':
        label_encoder=preprocessing.LabelEncoder()
        label_encoder.fit(list(processed_join_test[column_name].values))
        processed_join_test[column_name]=label_encoder.transform(list(processed_join_test[column_name].values))

processed_join_test.fillna((-999),inplace=True)
# print(processed_join_test.tail())

testing_data_input=processed_join_test.drop(['user_id'],axis=1)
testing_data_input=testing_data_input.drop(['business_id'],axis=1)
testing_data_input=testing_data_input.drop(['stars'],axis=1)
testing_data_input=testing_data_input.values

# print(testing_data_input[:10])

resulting_predictions=ratings_predictor_model.predict(data=testing_data_input)
print(testing_data_input.shape)
print(user_key_values_of_our_data.shape)
print(business_key_values_of_our_data.shape)

the_predictions=pd.DataFrame()
the_predictions['user_id']=user_key_values_of_our_data
the_predictions['business_id']=business_key_values_of_our_data
the_predictions['prediction']=resulting_predictions

print("RESULT_______________________________")

print(the_predictions.head())

the_predictions.to_csv(output_file_path,sep=',',encoding='utf-8',index=False)


#Time
print("Duration:",(time.time() - start))

"""
reference_file=open(testing_file_path,"r")
output_file=open(output_file_path,"r")

n=0
rmse=0

test_dict={}
output_dict={}

while(True):
    l1=output_file.readline()
    l2=reference_file.readline()
    if "user_id" in l2:
        continue
    if l2=="":
        break

    if not l1 and not l2:
        break
    #print(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    test_dict[(l2.split(",")[0],l2.split(",")[1])]=float(l2.split(",")[2][:-1])
    output_dict[(l1.split(",")[0], l1.split(",")[1])] = float(l1.split(",")[2][:-1])

for k in output_dict:
    n+=1
    rmse += math.pow(output_dict[k] - test_dict[k], 2)

rmse=math.sqrt(rmse/n)

print(rmse)
"""

