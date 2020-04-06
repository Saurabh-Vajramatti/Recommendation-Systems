from pyspark import SparkConf, SparkContext
import sys

import time

def hash_value(hash_index,user_code):
    return (hash_index*user_code +1)%number_of_users

def create_hash_list(user_code_list):

    hash_list=[]
    hash_list_length=number_of_hash_functions
    while(hash_list_length!=0):
        hash_list.append(float("inf"))
        hash_list_length=hash_list_length-1

    for user_code in user_code_list:
        for i in range(0,number_of_hash_functions):
            if(hash_value(i,user_code)<hash_list[i]):
                hash_list[i]=hash_value(i,user_code)

    return hash_list

def execute_banding(business_id,hash_user_list):

    band_number=0
    business_bands = []
    while(band_number!=number_of_bands):
        current_band_contents=hash_user_list[band_number*number_of_rows_in_one_band:band_number*number_of_rows_in_one_band+number_of_rows_in_one_band]
        current_band_contents.append(band_number)
        business_band_tuple=(tuple(current_band_contents),{business_id})
        business_bands.append(business_band_tuple)
        band_number=band_number+1

    return business_bands


def get_final_pairs(business_ids_set):

    final_pairs_list=[]
    business_ids_list=list(business_ids_set)
    for i in range(0,len(business_ids_list)):
        for j in range(i+1,len(business_ids_list)):
            business_id_one=business_ids_list[i]
            business_id_two=business_ids_list[j]
            intersection_count=len(business_dictionary[business_id_one].intersection(business_dictionary[business_id_two]))
            union_count=len(business_dictionary[business_id_one].union(business_dictionary[business_id_two]))
            if(intersection_count/union_count>=0.5):
                business_pair_list=[]
                business_pair_list.append(business_id_one)
                business_pair_list.append(business_id_two)
                business_pair_tuple=tuple(sorted(business_pair_list))
                business_pair_with_similarity_tuple=(business_pair_tuple,intersection_count/union_count)
                final_pairs_list.append(business_pair_with_similarity_tuple)

    return final_pairs_list

start = time.time()

sc = SparkContext(conf=SparkConf().setAppName("task1").setMaster("local[*]"))
sc.setLogLevel("OFF")

input_file_path=sys.argv[1]
output_file_path=sys.argv[2]

file_rdd=sc.textFile(input_file_path)

head=file_rdd.first()
file_rdd=file_rdd.filter(lambda l:l!=head)

file_rdd=file_rdd.map(lambda line: line.split(","))
#print(file_rdd.first())
file_rdd.persist()

number_of_users=file_rdd.map(lambda line:line[0]).distinct().count()
number_of_bands=45
number_of_rows_in_one_band=2
number_of_hash_functions=90
user_id_dictionary=file_rdd.map(lambda line:line[0]).distinct().zipWithIndex().collectAsMap()

# for u in user_id_dictionary:
#     print(user_id_dictionary[u])

rdd = file_rdd.map(lambda l: (l[1],{user_id_dictionary[l[0]]}))
business_baskets = rdd.reduceByKey(lambda a, b: a.union(b))
business_baskets.persist()


business_baskets_collected=business_baskets.collect()
business_dictionary={}

for business in business_baskets_collected:
    business_dictionary[business[0]]=business[1]

hash_table=business_baskets.map(lambda l:(l[0],create_hash_list(list(l[1]))))

banded_rdd=hash_table.flatMap(lambda l: execute_banding(l[0],l[1]))

band_commonality_rdd=banded_rdd.reduceByKey(lambda a,b: a.union(b)).filter(lambda l: len(l[1])>1)

final_pairs_rdd=band_commonality_rdd.flatMap(lambda l: get_final_pairs(l[1])).distinct().sortByKey().collect()

#print(final_pairs_rdd)

#print(band_commonality_rdd.first())

#print(banded_rdd.first())


#print(hash_table.first())


#print(hash_table.collect())



# output_file=open(output_file_path,"w")
# output_file.write("business_id_1, business_id_2, similarity\n")
#
# for business_pair_record in final_pairs_rdd:
#     output_file.write(business_pair_record[0][0]+","+business_pair_record[0][1]+","+str(business_pair_record[1]))
#     if(business_pair_record!=final_pairs_rdd[-1]):
#         output_file.write("\n")


output_file=open(output_file_path,"w")
output_file.write("business_id_1, business_id_2, similarity\n")

for business_pair_record in final_pairs_rdd:
    output_file.write(business_pair_record[0][0]+","+business_pair_record[0][1]+","+str(business_pair_record[1]))
    if(business_pair_record!=final_pairs_rdd[-1]):
        output_file.write("\n")


"""
#Validating accuracy

output_file.close()
reference_file=open("pure_jaccard_similarity.csv","r")
output_file=open(output_file_path,"r")
truthfile = open("pure_jaccard_similarity.csv","r")
true_positives = 0
total_myfile=0
for line in output_file:
    total_myfile+=1
    truthfile = open("pure_jaccard_similarity.csv", "r")
    for tline in truthfile:
        if(line==tline):
            true_positives+=1
truthfile = open("pure_jaccard_similarity.csv","r")
total_truth_file = 0
for line in truthfile:
    total_truth_file+=1
precision = true_positives/total_myfile
recall = true_positives/total_truth_file
print(precision)
print(recall)



#Time
print("Duration:",(time.time() - start))

"""