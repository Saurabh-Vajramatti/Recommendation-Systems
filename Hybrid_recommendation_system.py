from pyspark import SparkConf, SparkContext
import sys
import math

import time





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


neighbour_count_dictionary={}

sc = SparkContext(conf=SparkConf().setAppName("task1").setMaster("local[*]"))
sc.setLogLevel("OFF")

def item_based_CF():
    business_to_similarity_dictionary = {}

    def predict_for_a_pair(user_id, business_id):
        # if((user_id not in user_to_business_training_dictionary) or (business_id not in business_to_user_training_dictionary)):
        #     return ((user_id,business_id),2.5)

        # its 11270
        total_number_of_users = len(user_to_business_training_dictionary)
        if ((user_id not in user_to_business_training_dictionary) and (business_id in business_to_user_training_dictionary)):
            neighbour_count_dictionary[(user_id,business_id)] = 0
            return ((user_id, business_id), business_to_sumCount_training_dictionary[business_id][0] /business_to_sumCount_training_dictionary[business_id][1])
        elif ((user_id in user_to_business_training_dictionary) and (business_id not in business_to_user_training_dictionary)):
            neighbour_count_dictionary[(user_id,business_id)] = 0
            return ((user_id, business_id),user_to_sumCount_training_dictionary[user_id][0] / user_to_sumCount_training_dictionary[user_id][1])
        elif ((user_id not in user_to_business_training_dictionary) and (business_id not in business_to_user_training_dictionary)):
            print("BOTH NEW-------------------------------------------------------------------------->>")
            neighbour_count_dictionary[(user_id,business_id)] = 0
            return ((user_id, business_id), 2.5)

        businesses_rated_by_user = user_to_business_training_dictionary[user_id]

        neighbour_count_dictionary[(user_id,business_id)]=len(businesses_rated_by_user)
        # if(user_id=='6P7o2yz4-B8LZDgoqU_Cgg'):
        #     print("ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
        #     if('RCsid-VHYQI4UHWEII7s4Q' in businesses_rated_by_user):
        #         print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        if (business_id in businesses_rated_by_user):
            return ((user_id, business_id), userAndBusiness_to_rating_training_dictionary[(user_id, business_id)])

        # AVG of all method
        active_business_item_average = business_to_sumCount_training_dictionary[business_id][0] / \
                                       business_to_sumCount_training_dictionary[business_id][1]
        active_business_item_users = business_to_user_training_dictionary[business_id]

        current_businesses_similarity_dictionary = {}
        for business_item in businesses_rated_by_user:
            if (tuple(sorted([business_id, business_item])) in business_to_similarity_dictionary):
                current_businesses_similarity_dictionary[business_item] = business_to_similarity_dictionary[
                    tuple(sorted([business_id, business_item]))]
                continue
            else:
                # AVG of all method
                business_item_average = business_to_sumCount_training_dictionary[business_item][0] / \
                                        business_to_sumCount_training_dictionary[business_item][1]

                # Selected AVG method
                business_item_users = business_to_user_training_dictionary[business_item]
                common_users_set = business_item_users.intersection(active_business_item_users)

                # print(len(common_users_set))
                if (len(common_users_set) == 0):
                    # print("=======================================================>")
                    if (active_business_item_average / business_item_average > 1):
                        no_common_users_similarity = 1 / (active_business_item_average / business_item_average)
                    else:
                        no_common_users_similarity = active_business_item_average / business_item_average

                    current_businesses_similarity_dictionary[business_item] = business_to_similarity_dictionary[
                        tuple(sorted([business_id, business_item]))] = no_common_users_similarity
                    continue
                # #Finding common user average for each business to be used for default voting
                # business_item_average_on_common_users=0
                # active_business_item_average_on_common_users=0
                # for common_user in common_users_set:
                #     business_item_average_on_common_users+=userAndBusiness_to_rating_training_dictionary[(common_user,business_item)]
                #     active_business_item_average_on_common_users+=userAndBusiness_to_rating_training_dictionary[(common_user,business_id)]
                #
                # business_item_average_on_common_users/=len(common_users_set)
                # active_business_item_average_on_common_users/=len(common_users_set)
                # # Finding common user average for each business to be used for default voting END

                similarity_numerator = 0
                similarity_denominator_part1 = 0
                similarity_denominator_part2 = 0

                # #Default voting by chee method
                # for user in business_to_user_training_dictionary[business_item].union(business_to_user_training_dictionary[business_id]):
                #     if((user,business_item) not in userAndBusiness_to_rating_training_dictionary):
                #         business_item_numerator_contribution=business_item_average_on_common_users-business_item_average
                #     else:
                #         business_item_numerator_contribution=userAndBusiness_to_rating_training_dictionary[(user,business_item)]-business_item_average
                #
                #     if ((user, business_id) not in userAndBusiness_to_rating_training_dictionary):
                #         active_business_item_numerator_contribution = active_business_item_average_on_common_users - active_business_item_average
                #     else:
                #         active_business_item_numerator_contribution=userAndBusiness_to_rating_training_dictionary[(user,business_id)]-active_business_item_average
                #
                #     similarity_numerator+=(business_item_numerator_contribution)*(active_business_item_numerator_contribution)
                #     similarity_denominator_part1+=math.pow(business_item_numerator_contribution,2)
                #     similarity_denominator_part2+=math.pow(active_business_item_numerator_contribution,2)

                # Common user approach
                for user in common_users_set:
                    business_item_numerator_contribution = userAndBusiness_to_rating_training_dictionary[
                                                               (user, business_item)] - business_item_average
                    active_business_item_numerator_contribution = userAndBusiness_to_rating_training_dictionary[(
                        user, business_id)] - active_business_item_average
                    similarity_numerator += (business_item_numerator_contribution) * (
                        active_business_item_numerator_contribution)
                    similarity_denominator_part1 += math.pow(business_item_numerator_contribution, 2)
                    similarity_denominator_part2 += math.pow(active_business_item_numerator_contribution, 2)

                # print(math.sqrt(similarity_denominator_part2),math.sqrt(similarity_denominator_part1))
                if (similarity_numerator == 0):
                    pearson_similarity = 0
                else:
                    pearson_similarity = similarity_numerator / (
                            math.sqrt(similarity_denominator_part2) * math.sqrt(similarity_denominator_part1))
                business_to_similarity_dictionary[tuple(sorted([business_item, business_id]))] = pearson_similarity
                # if(len(common_users_set)<5):
                #     business_to_similarity_dictionary[tuple(sorted([business_item,business_id]))]/=2
                current_businesses_similarity_dictionary[business_item] = business_to_similarity_dictionary[
                    tuple(sorted([business_item, business_id]))]
                # if(pearson_similarity<-1 or pearson_similarity>1):
                #     print(pearson_similarity)

        prediction_numerator = 0
        prediction_denominator = 0

        # Case Amplification
        for considered_business in current_businesses_similarity_dictionary:
            if (current_businesses_similarity_dictionary[considered_business] > 0):
                current_businesses_similarity_dictionary[considered_business] = business_to_similarity_dictionary[
                    tuple(sorted([considered_business, business_id]))] = math.pow(
                    current_businesses_similarity_dictionary[considered_business], 6)

        businesses_sorted_by_similarity = []
        for k, v in sorted(current_businesses_similarity_dictionary.items(), key=lambda item: item[1], reverse=True):
            businesses_sorted_by_similarity.append(k)

        number_of_neighbours_allowed = len(businesses_sorted_by_similarity)
        for business_item in businesses_sorted_by_similarity:
            # print(current_businesses_similarity_dictionary[business_item])
            # print(business_to_similarity_dictionary[business_pair_tuple])

            if (current_businesses_similarity_dictionary[business_item] < 0):
                continue

            business_pair_tuple = tuple(sorted([business_item, business_id]))
            # if(business_to_similarity_dictionary[business_pair_tuple]<0):
            #     break
            number_of_neighbours_allowed -= 1

            prediction_numerator += userAndBusiness_to_rating_training_dictionary[(user_id, business_item)] * \
                                    current_businesses_similarity_dictionary[business_item]
            prediction_denominator += abs(current_businesses_similarity_dictionary[business_item])
            if (number_of_neighbours_allowed == 0):
                # print("END===========================>")
                break

        if (prediction_numerator == 0):
            predicted_rating = 0
        else:
            predicted_rating = prediction_numerator / prediction_denominator

        if (predicted_rating == 0):
            # print(("STILLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"))
            business_avg = business_to_sumCount_training_dictionary[business_id][0] / \
                           business_to_sumCount_training_dictionary[business_id][1]
            user_avg = user_to_sumCount_training_dictionary[user_id][0] / user_to_sumCount_training_dictionary[user_id][
                1]

            # print(user_id,business_id,business_avg,user_avg,(business_avg+user_avg)/2)
            return ((user_id, business_id), (business_avg + user_avg) / 2)

        if (predicted_rating > 5):
            predicted_rating = 5.0

        # predicted_rating=round(predicted_rating)
        return ((user_id, business_id), predicted_rating)

    start = time.time()

    training_file_path = sys.argv[1]+"/yelp_train.csv"
    testing_file_path = sys.argv[2]
    output_file_path = "itemBased.csv"

    # Training file
    training_file_rdd = sc.textFile(training_file_path)
    head = training_file_rdd.first()
    training_file_rdd = training_file_rdd.filter(lambda l: l != head)

    training_file_rdd = training_file_rdd.map(lambda line: line.split(","))
    # print(file_rdd.first())
    training_file_rdd.persist()

    # Testing file
    testing_file_rdd = sc.textFile(testing_file_path)
    head = testing_file_rdd.first()
    testing_file_rdd = testing_file_rdd.filter(lambda l: l != head)

    testing_file_rdd = testing_file_rdd.map(lambda line: line.split(","))
    # print(file_rdd.first())
    testing_file_rdd.persist()

    user_to_business_training_dictionary = training_file_rdd.map(lambda l: (l[0], {l[1]})).reduceByKey(
        lambda a, b: a.union(b)).collectAsMap()
    business_to_user_training_dictionary = training_file_rdd.map(lambda l: (l[1], {l[0]})).reduceByKey(
        lambda a, b: a.union(b)).collectAsMap()
    userAndBusiness_to_rating_training_dictionary = training_file_rdd.map(
        lambda l: ((l[0], l[1]), float(l[2]))).collectAsMap()
    business_to_sumCount_training_dictionary = training_file_rdd.map(lambda l: (l[1], (float(l[2]), 1))).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])).collectAsMap()
    user_to_sumCount_training_dictionary = training_file_rdd.map(lambda l: (l[0], (float(l[2]), 1))).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])).collectAsMap()
    # if('RCsid-VHYQI4UHWEII7s4Q' in user_to_business_training_dictionary['6P7o2yz4-B8LZDgoqU_Cgg']):
    #     print("YESSSSSSSSSSSSSSSSSSSSSSS")
    #
    # print(user_to_business_training_dictionary['6P7o2yz4-B8LZDgoqU_Cgg'])
    # print(business_to_sumCount_training_dictionary["5j7BnXXvlS69uLVHrY9Upw"])

    predicted_pairs = testing_file_rdd.map(lambda l: predict_for_a_pair(l[0], l[1])).collect()
    # initial_predicted_pairs=testing_file_rdd.map(lambda l: predict_for_a_pair(l[0],l[1])).persist()
    #
    # max_rating = initial_predicted_pairs.map(lambda entry: entry[1]).max()
    # min_rating = initial_predicted_pairs.map(lambda entry: entry[1]).min()
    # max_min_diff = max_rating - min_rating
    #
    # print("Min: ", min_rating)
    # print("Max: ", max_rating)
    # print("Diff: ", max_min_diff)
    #
    # predicted_pairs = initial_predicted_pairs\
    #     .mapValues(lambda entry: ((entry - min_rating) / max_min_diff) * 4 + 1)
    #
    # predicted_pairs=predicted_pairs.collect()

    output_file = open(output_file_path, "w")
    output_file.write("user_id, business_id, prediction\n")

    for predicted_record in predicted_pairs:
        output_file.write(predicted_record[0][0] + "," + predicted_record[0][1] + "," + str(predicted_record[1]))
        if (predicted_record != predicted_pairs[-1]):
            output_file.write("\n")

    # Time
    print("Duration:", (time.time() - start))

    # # Validating accuracy
    #
    # # output_file.close()
    # reference_file = open(testing_file_path, "r")
    # output_file = open(output_file_path, "r")
    #
    # n = 0
    # rmse = 0
    #
    # while (True):
    #     l1 = output_file.readline()
    #     l2 = reference_file.readline()
    #     if "user_id" in l2:
    #         continue
    #     if l2 == "":
    #         break
    #     n += 1
    #     if (n == 1):
    #         print(float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    #     # print(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    #     rmse += math.pow(float(l1.split(",")[2][:-1]) - float(l2.split(",")[2][:-1]), 2)
    #     # print(rmse)
    #     if not l1 and not l2:
    #         break
    #
    # rmse = math.sqrt(rmse / n)
    #
    # print(rmse)


def xgboost_model():

    start = time.time()

    training_folder_path = sys.argv[1]
    testing_file_path = sys.argv[2]
    output_file_path = "xgboostBased.csv"

    user_file = pd.read_json(training_folder_path + "/user.json", lines=True, chunksize=100000)
    for user_chunk in user_file:
        first_user_data_chunk = user_chunk
        break

    all_user_data = first_user_data_chunk.copy()


    flag = 0
    for user_chunk in user_file:
        all_user_data = all_user_data.append(user_chunk)

    print(all_user_data.shape)
    print(all_user_data.head())

    business_file = pd.read_json(training_folder_path + "/business.json", lines=True, chunksize=100000)
    for business_chunk in business_file:
        first_business_data_chunk = business_chunk
        break

    all_business_data = first_business_data_chunk.copy()

    flag = 0
    for business_chunk in business_file:
        all_business_data = all_business_data.append(business_chunk)

    #
    # print(first_user_data_chunk.shape)
    # print(first_user_data_chunk.dtypes)
    # print(first_user_data_chunk.head(5))

    train_set = pd.read_csv(training_folder_path + "/yelp_train.csv")
    # print(train_set.head())

    data_required_from_user = all_user_data[['user_id', 'average_stars', 'review_count', 'useful', 'fans','funny','cool','compliment_hot','compliment_more','compliment_profile','compliment_cute','compliment_list','compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos']]
    print(data_required_from_user.dtypes)
    data_required_from_business = all_business_data[['business_id', 'stars', 'review_count']]
    data_required_from_business = data_required_from_business.rename(
        columns={'stars': 'business_avg_stars', 'review_count': 'business_rev_count'})
    print(data_required_from_business.dtypes)

    print(data_required_from_business.head())

    join_of_requiredUserData_and_train = pd.merge(train_set, data_required_from_user, on='user_id', how='inner')
    join_of_requiredUserData_and_train = pd.merge(join_of_requiredUserData_and_train, data_required_from_business,
                                                  on='business_id', how='inner')
    # print(join_of_requiredUserData_and_train[['user_id','business_id','review_count']].head(30))

    processed_join_train = join_of_requiredUserData_and_train.copy()

    for column_name in processed_join_train.columns:
        if processed_join_train[column_name].dtype == 'object':
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(list(processed_join_train[column_name].values))
            processed_join_train[column_name] = label_encoder.transform(list(processed_join_train[column_name].values))

    # print(processed_join_train.tail())

    training_data_result_ratings = processed_join_train.stars.values
    training_data_input = processed_join_train.drop(['stars'], axis=1)
    training_data_input = training_data_input.drop(['user_id'], axis=1)
    training_data_input = training_data_input.drop(['business_id'], axis=1)
    training_data_input = training_data_input.values

    ratings_predictor_model = xgb.XGBRegressor()
    ratings_predictor_model.fit(training_data_input, training_data_result_ratings)
    # print(training_data_input[:10])
    # print(training_data_result_ratings[:10])

    # print(ratings_predictor_model)

    test_set = pd.read_csv(testing_file_path)

    join_of_requiredUserData_and_test = pd.merge(test_set, data_required_from_user, on='user_id', how='left')
    join_of_requiredUserData_and_test = pd.merge(join_of_requiredUserData_and_test, data_required_from_business,
                                                 on='business_id', how='left')

    # print(join_of_requiredUserData_and_test[['user_id','business_id','review_count']].head(30))

    user_key_values_of_our_data = join_of_requiredUserData_and_test['user_id'].values
    business_key_values_of_our_data = join_of_requiredUserData_and_test['business_id'].values

    # print(user_key_values_of_our_data[:10])
    # print(business_key_values_of_our_data[:10])

    processed_join_test = join_of_requiredUserData_and_test.copy()

    for column_name in processed_join_test.columns:
        if processed_join_test[column_name].dtype == 'object':
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(list(processed_join_test[column_name].values))
            processed_join_test[column_name] = label_encoder.transform(list(processed_join_test[column_name].values))

    processed_join_test.fillna((-999), inplace=True)
    # print(processed_join_test.tail())

    testing_data_input = processed_join_test.drop(['user_id'], axis=1)
    testing_data_input = testing_data_input.drop(['business_id'], axis=1)
    testing_data_input = testing_data_input.drop(['stars'], axis=1)
    testing_data_input = testing_data_input.values

    # print(testing_data_input[:10])

    resulting_predictions = ratings_predictor_model.predict(data=testing_data_input)
    print(testing_data_input.shape)
    print(user_key_values_of_our_data.shape)
    print(business_key_values_of_our_data.shape)

    the_predictions = pd.DataFrame()
    the_predictions['user_id'] = user_key_values_of_our_data
    the_predictions['business_id'] = business_key_values_of_our_data
    the_predictions['prediction'] = resulting_predictions

    print("RESULT_______________________________")

    print(the_predictions.head())

    the_predictions.to_csv(output_file_path, sep=',', encoding='utf-8', index=False)

    # data1 = the_predictions.iloc[0:len(the_predictions) - 1]
    # data2 = the_predictions.iloc[[len(the_predictions) - 1]]
    # data1.to_csv(output_file_path, sep=',', encoding='utf-8', header=False, index=False)
    # data2.to_csv(output_file_path, sep=',', encoding='utf-8', header=False, index=False, mode='a', line_terminator="")
    # Time
    print("Duration:", (time.time() - start))

    # reference_file = open(testing_file_path, "r")
    # output_file = open(output_file_path, "r")
    #
    # n = 0
    # rmse = 0
    #
    # test_dict = {}
    # output_dict = {}
    #
    # while (True):
    #     l1 = output_file.readline()
    #     l2 = reference_file.readline()
    #     if "user_id" in l2:
    #         continue
    #     if l2 == "":
    #         break
    #
    #     if not l1 and not l2:
    #         break
    #     # print(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    #     test_dict[(l2.split(",")[0], l2.split(",")[1])] = float(l2.split(",")[2][:-1])
    #     output_dict[(l1.split(",")[0], l1.split(",")[1])] = float(l1.split(",")[2][:-1])
    #
    # for k in output_dict:
    #     n += 1
    #     rmse += math.pow(output_dict[k] - test_dict[k], 2)
    #
    # rmse = math.sqrt(rmse / n)
    #
    # print(rmse)


start = time.time()

final_training_folder_path=sys.argv[1]
final_testing_file_path=sys.argv[2]
final_output_file_path=sys.argv[3]

test_file_temp=open(final_testing_file_path,"r")
test_file_length=0
for line in test_file_temp:
    if(line!=""):
        break
    test_file_length+=1

hybrid_file=open(final_output_file_path,"w")

item_based_CF()

xgboost_model()




testing_file_path = sys.argv[2]
# Testing file
testing_file_rdd = sc.textFile(testing_file_path)
head = testing_file_rdd.first()
testing_file_rdd = testing_file_rdd.filter(lambda l: l != head)

testing_file_rdd = testing_file_rdd.map(lambda line: line.split(",")).map(lambda l:(l[0],l[1])).collect()

# Training file
training_file_rdd = sc.textFile(final_training_folder_path+"/yelp_train.csv")
head = training_file_rdd.first()
training_file_rdd = training_file_rdd.filter(lambda l: l != head)

training_file_rdd = training_file_rdd.map(lambda line: line.split(","))
# print(file_rdd.first())
training_file_rdd.persist()

user_to_business_training_dictionary = training_file_rdd.map(lambda l: (l[0], {l[1]})).reduceByKey(lambda a, b: a.union(b)).collectAsMap()

for predicted_record in testing_file_rdd:
    neighbour_count_dictionary[predicted_record]=len(user_to_business_training_dictionary[predicted_record[0]])

print(len(neighbour_count_dictionary))

item_file=open("itemBased.csv","r")
boost_file=open("xgboostBased.csv","r")

max_neighbour_count=0
for pair in neighbour_count_dictionary:
    if(max_neighbour_count<neighbour_count_dictionary[pair]):
        max_neighbour_count=neighbour_count_dictionary[pair]


line_counter=0

hybrid_file.write("user_id, business_id, prediction\n")
while(True):
    item_file_line=item_file.readline()
    boost_file_line=boost_file.readline()

    line_counter+=1
    if "user_id" in item_file_line:
        continue

    if boost_file_line=="":
        break
    if not item_file_line and not boost_file_line:
        break

    item_user=item_file_line.split(",")[0]
    item_business=item_file_line.split(",")[1]
    if(item_file_line.split(",")[-1]=="\n"):
        itemBased_rating=float(item_file_line.split(",")[2][:-1])
    else:
        itemBased_rating = float(item_file_line.split(",")[2])

    boost_user = boost_file_line.split(",")[0]
    boost_business = boost_file_line.split(",")[1]
    if (boost_file_line.split(",")[-1] == "\n"):
        boostBased_rating = float(boost_file_line.split(",")[2][:-1])
    else:
        boostBased_rating = float(boost_file_line.split(",")[2])


    item_based_weight=(neighbour_count_dictionary[(item_user,item_business)]/max_neighbour_count)
    # if(item_based_weight<=0.3):
    #     item_based_weight+=0.2

    # item_based_weight=1
    boost_based_weight=1-item_based_weight
    final_hybrid_rating=item_based_weight*itemBased_rating +boost_based_weight*boostBased_rating

    hybrid_file.write(item_user+","+item_business+","+str(final_hybrid_rating))
    if(line_counter!=test_file_length):
        hybrid_file.write("\n")




#Time
print("Duration:",(time.time() - start))







"""
reference_file=open(final_testing_file_path,"r")
output_file=open(final_output_file_path,"r")

n=0
rmse=0

test_dict={}
output_dict={}

while(True):
    l1=output_file.readline()
    l2=reference_file.readline()
    if "user_id" in l2:
        continue
    if l2=="" or l1=="":
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