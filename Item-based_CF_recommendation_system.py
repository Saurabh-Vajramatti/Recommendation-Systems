from pyspark import SparkConf, SparkContext
import sys
import math

import time


business_to_similarity_dictionary={}
def predict_for_a_pair(user_id,business_id):
    # if((user_id not in user_to_business_training_dictionary) or (business_id not in business_to_user_training_dictionary)):
    #     return ((user_id,business_id),2.5)

    #its 11270
    total_number_of_users = len(user_to_business_training_dictionary)
    if((user_id not in user_to_business_training_dictionary) and (business_id in business_to_user_training_dictionary)):
        return ((user_id,business_id),business_to_sumCount_training_dictionary[business_id][0]/business_to_sumCount_training_dictionary[business_id][1])
    elif((user_id in user_to_business_training_dictionary) and (business_id not in business_to_user_training_dictionary)):
        return ((user_id,business_id),user_to_sumCount_training_dictionary[user_id][0]/user_to_sumCount_training_dictionary[user_id][1])
    elif((user_id not in user_to_business_training_dictionary) and (business_id not in business_to_user_training_dictionary)):
        print("BOTH NEW-------------------------------------------------------------------------->>")
        return ((user_id, business_id), 2.5)

    businesses_rated_by_user = user_to_business_training_dictionary[user_id]

    # if(user_id=='6P7o2yz4-B8LZDgoqU_Cgg'):
    #     print("ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
    #     if('RCsid-VHYQI4UHWEII7s4Q' in businesses_rated_by_user):
    #         print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    if(business_id in businesses_rated_by_user):
        return ((user_id,business_id),userAndBusiness_to_rating_training_dictionary[(user_id,business_id)])




    # AVG of all method
    active_business_item_average=business_to_sumCount_training_dictionary[business_id][0]/business_to_sumCount_training_dictionary[business_id][1]
    active_business_item_users=business_to_user_training_dictionary[business_id]

    current_businesses_similarity_dictionary={}
    for business_item in businesses_rated_by_user:
        if(tuple(sorted([business_id,business_item])) in business_to_similarity_dictionary):
            current_businesses_similarity_dictionary[business_item]=business_to_similarity_dictionary[tuple(sorted([business_id,business_item]))]
            continue
        else:
            #AVG of all method
            business_item_average=business_to_sumCount_training_dictionary[business_item][0]/business_to_sumCount_training_dictionary[business_item][1]

            # Selected AVG method
            business_item_users=business_to_user_training_dictionary[business_item]
            common_users_set=business_item_users.intersection(active_business_item_users)

            #print(len(common_users_set))
            if (len(common_users_set) == 0):
                # print("=======================================================>")
                if(active_business_item_average/business_item_average>1):
                    no_common_users_similarity=1/(active_business_item_average/business_item_average)
                else:
                    no_common_users_similarity = active_business_item_average/business_item_average

                current_businesses_similarity_dictionary[business_item] = business_to_similarity_dictionary[tuple(sorted([business_id, business_item]))]=no_common_users_similarity
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



            similarity_numerator=0
            similarity_denominator_part1=0
            similarity_denominator_part2=0



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

            #Common user approach
            for user in common_users_set:
                business_item_numerator_contribution=userAndBusiness_to_rating_training_dictionary[(user,business_item)]-business_item_average
                active_business_item_numerator_contribution=userAndBusiness_to_rating_training_dictionary[(user,business_id)]-active_business_item_average
                similarity_numerator+=(business_item_numerator_contribution)*(active_business_item_numerator_contribution)
                similarity_denominator_part1+=math.pow(business_item_numerator_contribution,2)
                similarity_denominator_part2+=math.pow(active_business_item_numerator_contribution,2)

            # print(math.sqrt(similarity_denominator_part2),math.sqrt(similarity_denominator_part1))
            if(similarity_numerator==0):
                pearson_similarity=0
            else:
                pearson_similarity=similarity_numerator/(math.sqrt(similarity_denominator_part2)*math.sqrt(similarity_denominator_part1))
            business_to_similarity_dictionary[tuple(sorted([business_item,business_id]))]=pearson_similarity
            # if(len(common_users_set)<5):
            #     business_to_similarity_dictionary[tuple(sorted([business_item,business_id]))]/=2
            current_businesses_similarity_dictionary[business_item]=business_to_similarity_dictionary[tuple(sorted([business_item,business_id]))]
            # if(pearson_similarity<-1 or pearson_similarity>1):
            #     print(pearson_similarity)

    prediction_numerator=0
    prediction_denominator=0

    # Case Amplification
    for considered_business in current_businesses_similarity_dictionary:
        if(current_businesses_similarity_dictionary[considered_business]>0):
            current_businesses_similarity_dictionary[considered_business] = business_to_similarity_dictionary[tuple(sorted([considered_business, business_id]))]=math.pow(current_businesses_similarity_dictionary[considered_business],2.5)

    businesses_sorted_by_similarity=[]
    for k, v in sorted(current_businesses_similarity_dictionary.items(), key=lambda item: item[1],reverse=True):
        businesses_sorted_by_similarity.append(k)

    number_of_neighbours_allowed=len(businesses_sorted_by_similarity)
    for business_item in businesses_sorted_by_similarity:
        # print(current_businesses_similarity_dictionary[business_item])
        # print(business_to_similarity_dictionary[business_pair_tuple])


        if(current_businesses_similarity_dictionary[business_item]<0):
            continue

        business_pair_tuple=tuple(sorted([business_item,business_id]))
        # if(business_to_similarity_dictionary[business_pair_tuple]<0):
        #     break
        number_of_neighbours_allowed-=1

        prediction_numerator+=userAndBusiness_to_rating_training_dictionary[(user_id,business_item)]*current_businesses_similarity_dictionary[business_item]
        prediction_denominator+=abs(current_businesses_similarity_dictionary[business_item])
        if(number_of_neighbours_allowed==0):
            # print("END===========================>")
            break

    if(prediction_numerator==0):
        predicted_rating=0
    else:
        predicted_rating=prediction_numerator/prediction_denominator

    if(predicted_rating==0):
        # print(("STILLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"))
        business_avg=business_to_sumCount_training_dictionary[business_id][0]/business_to_sumCount_training_dictionary[business_id][1]
        user_avg=user_to_sumCount_training_dictionary[user_id][0]/user_to_sumCount_training_dictionary[user_id][1]

        # print(user_id,business_id,business_avg,user_avg,(business_avg+user_avg)/2)
        return ((user_id, business_id),(business_avg+user_avg)/2)

    if(predicted_rating>5):
        predicted_rating=5.0

    # predicted_rating=round(predicted_rating)
    return ((user_id,business_id),predicted_rating)


start = time.time()

sc = SparkContext(conf=SparkConf().setAppName("task1").setMaster("local[*]"))
sc.setLogLevel("OFF")

training_file_path=sys.argv[1]
testing_file_path=sys.argv[2]
output_file_path=sys.argv[3]

#Training file
training_file_rdd=sc.textFile(training_file_path)
head=training_file_rdd.first()
training_file_rdd=training_file_rdd.filter(lambda l:l!=head)

training_file_rdd=training_file_rdd.map(lambda line: line.split(","))
#print(file_rdd.first())
training_file_rdd.persist()


#Testing file
testing_file_rdd=sc.textFile(testing_file_path)
head=testing_file_rdd.first()
testing_file_rdd=testing_file_rdd.filter(lambda l:l!=head)

testing_file_rdd=testing_file_rdd.map(lambda line: line.split(","))
#print(file_rdd.first())
testing_file_rdd.persist()

user_to_business_training_dictionary=training_file_rdd.map(lambda l:(l[0],{l[1]})).reduceByKey(lambda a,b:a.union(b)).collectAsMap()
business_to_user_training_dictionary=training_file_rdd.map(lambda l:(l[1],{l[0]})).reduceByKey(lambda a,b:a.union(b)).collectAsMap()
userAndBusiness_to_rating_training_dictionary=training_file_rdd.map(lambda l:((l[0],l[1]),float(l[2]))).collectAsMap()
business_to_sumCount_training_dictionary=training_file_rdd.map(lambda l:(l[1],(float(l[2]),1))).reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1])).collectAsMap()
user_to_sumCount_training_dictionary=training_file_rdd.map(lambda l:(l[0],(float(l[2]),1))).reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1])).collectAsMap()
# if('RCsid-VHYQI4UHWEII7s4Q' in user_to_business_training_dictionary['6P7o2yz4-B8LZDgoqU_Cgg']):
#     print("YESSSSSSSSSSSSSSSSSSSSSSS")
#
# print(user_to_business_training_dictionary['6P7o2yz4-B8LZDgoqU_Cgg'])
# print(business_to_sumCount_training_dictionary["5j7BnXXvlS69uLVHrY9Upw"])



predicted_pairs=testing_file_rdd.map(lambda l: predict_for_a_pair(l[0],l[1])).collect()
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

output_file=open(output_file_path,"w")
output_file.write("user_id, business_id, prediction\n")

for predicted_record in predicted_pairs:
    output_file.write(predicted_record[0][0]+","+predicted_record[0][1]+","+str(predicted_record[1]))
    if(predicted_record!=predicted_pairs[-1]):
        output_file.write("\n")


"""
#Time
print("Duration:",(time.time() - start))


#Validating accuracy

# output_file.close()
reference_file=open(testing_file_path,"r")
output_file=open(output_file_path,"r")

n=0
rmse=0


while(True):
    l1=output_file.readline()
    l2=reference_file.readline()
    if "user_id" in l2:
        continue
    if l2=="":
        break
    n+=1
    if(n==1):
        print(float(l1.split(",")[2][:-1]),float(l2.split(",")[2][:-1]))
    #print(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    rmse+=math.pow(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),2)
    #print(rmse)
    if not l1 and not l2:
        break

rmse=math.sqrt(rmse/n)

print(rmse)

"""
