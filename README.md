# Recommendation Systems
 Implementation of Locality Sensitive Hashing (LSH), and different types of collaborative-filtering recommendation systems

# Programming Environment
Python: 3.6.4
Spark: 2.4.4
JDK: 1.8 

# Generating dataset from Yelp Dataset
Generated the following two datasets from the original Yelp review dataset with some filters such as the condition: “state” == “CA”. Randomly took 60% of the data as the training dataset, 20% of the data as the validation dataset.

a. yelp_train.csv: the training data, which only include the columns: user_id, business_id, and stars.
b. yelp_val.csv: the validation data, which are in the same format as training data

# Running the algorithm
# Task1: Jaccard based LSH
Command to run:
spark-submit Jaccard_based_LSH.py <input_file_path> <output_file_path>

Implemented the Locality Sensitive Hashing algorithm with Jaccard similarity using yelp_train.csv with focus on the “0 or 1” ratings rather than the actual ratings/stars from the users. Specifically, if a user has rated a business, the user’s contribution in the characteristic matrix is 1. If the user hasn’t rated the business, the contribution is 0. Identified similar businesses whose similarity >=0.5. The generated results are compared to the ground truth file pure_jaccard_similarity.csv.

# Task2.1:  Item-based CF recommendation system 
# Command to run:
spark-submit Item-based_CF_recommendation_system.py <train_filename> <test_filename.py> <output_filename>

This implementation generates recommendations using Item-based CF recommendation system

# Task2.2:  Model-based recommendation system  
# Command to run:
spark-submit Model-based_recommendation_system.py  <folder_path> <test_file_name> <output_file_name> 
 *folder_path: the path of dataset folder, additional files like user.json and business.json

This implementation generates recommendations using Model-based CF recommendation system with the use of XGBregressor(a regressor based on the decision tree) to train a model.

# Task2.3:  Hybrid recommendation system  
# Command to run:
spark-submit Hybrid_recommendation_system.py  <folder_path> <test_file_name> <output_file_name> 
 *folder_path: the path of dataset folder, additional files like user.json and business.json

Here the goal was to combine the above "Item-based CF recommendation system" and "Model-based recommendation system" to design a better hybrid recommendation system. 

I went with combining them together as a weighted average, which means: 
                𝑓𝑖𝑛𝑎𝑙 𝑠𝑐𝑜𝑟𝑒 = 𝛼 × 𝑠𝑐𝑜𝑟𝑒𝑖𝑡𝑒𝑚_𝑏𝑎𝑠𝑒𝑑 + (1 − 𝛼) × 𝑠𝑐𝑜𝑟𝑒𝑚𝑜𝑑𝑒𝑙_𝑏𝑎𝑠𝑒𝑑

The key idea is: the CF focuses on the neighbors of the item and the model-based RS focuses on the user and item themselves. Specifically, if the item has a smaller number of neighbors, then the weight of the CF should be smaller.

# My RMSE value results:
Item-CF:1.068
Model:0.987
Hybrid:0.984