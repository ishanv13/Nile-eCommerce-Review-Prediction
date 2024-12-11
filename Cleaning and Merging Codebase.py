# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:01:51 2024

@author: Group 23
"""
#######################################################################################################################################

# Customers Dataset Cleaning

import pandas as pd
import numpy as np


df = pd.read_csv('olist_customers_dataset.csv')

print(df.isnull().sum())

print(df.head(15).to_string())

print(df.dtypes)


# Count of unique cities
print(len(df['customer_city'].unique()))

# Count of unique customers
print(len(df['customer_unique_id'].unique()))

# Count of unique states
print(len(df['customer_state'].unique()))

# Count of unique zipcodes
print(len(df['customer_zip_code_prefix'].unique()))


# Adding customer_order_frequency with transform
df['customer_order_frequency'] = df.groupby('customer_unique_id')['customer_id'].transform('nunique')

import seaborn as sns

# Plotting distributions of customer_order_frequency
grouped_df = df.groupby('customer_unique_id')['customer_id'].nunique().reset_index(name='customer_order_frequency')
sns.displot(grouped_df['customer_order_frequency'])
sns.boxenplot(grouped_df['customer_order_frequency'])

help(sns.boxenplot)

# Plotting distrubtion of cusomers with more than 2 orders
filtered_df = grouped_df[grouped_df['customer_order_frequency'] > 1]
sns.displot(filtered_df['customer_order_frequency'])

# Creating a new column for if orders are greater than 1 (i.e repeat customer)
df['multiple_orders'] = (df['customer_order_frequency'] > 1).astype(int)

# Getting rid of city (too many typos and errors)
df = df.drop(['customer_city'], axis = 1)
df.to_csv('customers_clean.csv', index=False)


#################################################################################################################################################

# Orders Data Cleaning

import pandas as pd
import numpy as np

"""Orders Dataset"""

orders_dataset=pd.read_csv('olist_orders_dataset.csv')

#Convert relevant columns to datetime format
orders_dataset['order_purchase_timestamp'] = pd.to_datetime(orders_dataset['order_purchase_timestamp'])
orders_dataset['order_approved_at'] = pd.to_datetime(orders_dataset['order_approved_at'])
orders_dataset['order_delivered_carrier_date'] = pd.to_datetime(orders_dataset['order_delivered_carrier_date'])
orders_dataset['order_delivered_customer_date'] = pd.to_datetime(orders_dataset['order_delivered_customer_date'])
orders_dataset['order_estimated_delivery_date'] = pd.to_datetime(orders_dataset['order_estimated_delivery_date'])

#Delete hours and seconds from all
orders_dataset['order_purchase_timestamp'] = orders_dataset['order_purchase_timestamp'].dt.date
orders_dataset['order_approved_at'] = orders_dataset['order_approved_at'].dt.date
orders_dataset['order_delivered_carrier_date'] = orders_dataset['order_delivered_carrier_date'].dt.date
orders_dataset['order_delivered_customer_date'] = orders_dataset['order_delivered_customer_date'].dt.date
orders_dataset['order_estimated_delivery_date'] = orders_dataset['order_estimated_delivery_date'].dt.date

#Create new column 'delivery_carrier_customer' to check difference between delivery to customers and delivery to carriers
orders_dataset['delivery_carrier_customer'] = orders_dataset['order_delivered_customer_date'] - orders_dataset['order_delivered_carrier_date']
#Create new column 'purchase_to_delivery' to check difference between when order was purchased and when order was delivered
orders_dataset['purchase_to_delivery'] = (orders_dataset['order_delivered_customer_date'] - orders_dataset['order_purchase_timestamp'])
#Create new column 'days_overdue' which tells the difference between the date orders were estimated to be delivered and when they were actually delivered to check for errors
orders_dataset['days_overdue'] = (orders_dataset['order_delivered_customer_date'] - orders_dataset['order_estimated_delivery_date'])

#Change new columns to timedelta
orders_dataset['delivery_carrier_customer'] = pd.to_timedelta(orders_dataset['delivery_carrier_customer'])
orders_dataset['purchase_to_delivery'] = pd.to_timedelta(orders_dataset['purchase_to_delivery'])
orders_dataset['days_overdue'] = pd.to_timedelta(orders_dataset['days_overdue'])

#Change new columns to days
orders_dataset['delivery_carrier_customer'] = orders_dataset['delivery_carrier_customer'].dt.days
orders_dataset['purchase_to_delivery'] = orders_dataset['purchase_to_delivery'].dt.days
orders_dataset['days_overdue'] = orders_dataset['days_overdue'].dt.days

#Check how many customer_ids there are and how many unique customer_ids there are (99441)
print(orders_dataset['customer_id'].count())
print(orders_dataset['customer_id'].nunique())

#Count how many 'delivery_carrier_customer' values are negative (20)
orders_dataset[orders_dataset['delivery_carrier_customer'] < 0].shape[0]
#Count null values (2966)
orders_dataset['delivery_carrier_customer'].isnull().sum()
#Count how many 'purchase_to_delivery' values are negative (0)
orders_dataset[orders_dataset['purchase_to_delivery'] < 0].shape[0]

#Keep everything that is not negative, so positive and null values
orders_dataset = orders_dataset[(orders_dataset['delivery_carrier_customer'] >= 0) | (orders_dataset['delivery_carrier_customer'].isnull())]
#count number of rows (99421)
orders_dataset.shape

#Change 'order_status' answers to 1 and 0, 1 represents 'delivered', 0 for all others
orders_dataset['order_status'] = orders_dataset['order_status'].apply(lambda x: 1 if x == 'delivered' else 0)
#Filter out all rows with '0' in the order_status column
orders_dataset = orders_dataset[orders_dataset['order_status'] == 1]
#count number of rows (96458)
orders_dataset.shape

#Drop used columns
orders_dataset.drop(['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date','order_estimated_delivery_date'], axis=1, inplace=True)

orders_dataset.head()

#download data
orders_dataset.to_csv('orders_dataset.csv', index=False)

###################################################################################################################################################################################################

# Order Payments Cleaning


import pandas as pd
import numpy as np

order_payments_dataset=pd.read_csv('olist_order_payments_dataset.csv')

#Check how many order_ids there are and how many unique order_ids there are
print(order_payments_dataset['order_id'].count()) #103886
print(order_payments_dataset['order_id'].nunique()) #99440

#Drop columns not used
order_payments_dataset.drop(['payment_sequential'], axis=1, inplace=True)

#Split each payment type in 'payment_type' into a seperate column, giving 1 if the row used this payment type and 0 if the row did not use this payment type
order_payments_dataset = order_payments_dataset.join(order_payments_dataset['payment_type'].str.get_dummies())

#Combine all non-distinct order_ids in order_payments_dataset, summing the remaining columns
order_payments_dataset = order_payments_dataset.groupby(['order_id']).agg({'payment_installments': 'sum', 'payment_value': 'sum', 'boleto': 'sum', 'credit_card': 'sum', 'debit_card': 'sum', 'not_defined': 'sum', 'voucher': 'sum'}).reset_index()

#Change all values above 1 in 'boleto, credit_card, debit_card, not_defined, and voucher to 1
order_payments_dataset['boleto'] = order_payments_dataset['boleto'].apply(lambda x: 1 if x > 1 else x)
order_payments_dataset['credit_card'] = order_payments_dataset['credit_card'].apply(lambda x: 1 if x > 1 else x)
order_payments_dataset['debit_card'] = order_payments_dataset['debit_card'].apply(lambda x: 1 if x > 1 else x)
order_payments_dataset['not_defined'] = order_payments_dataset['not_defined'].apply(lambda x: 1 if x > 1 else x)
order_payments_dataset['voucher'] = order_payments_dataset['voucher'].apply(lambda x: 1 if x > 1 else x)

#Create a new column which counts the number of unique payment types made for each unique order_id
order_payments_dataset['payment_type_count'] = order_payments_dataset['boleto'] + order_payments_dataset['credit_card']+order_payments_dataset['debit_card']+order_payments_dataset['not_defined']+order_payments_dataset['voucher']

#Rename each payment type with 'payment type' at the beginning for clarity
order_payments_dataset.rename(columns={'boleto': 'payment_type_boleto', 'credit_card': 'payment_type_credit_card', 'debit_card': 'payment_type_debit_card', 'not_defined': 'payment_type_not_defined', 'voucher': 'payment_type_voucher'}, inplace=True)

#Check how many rows there are (99440)
order_payments_dataset.shape

#Print first columns
order_payments_dataset.head()

order_payments_dataset.to_csv('order_payments_dataframe.csv', index=False)

###################################################################################################################################################################################################

# Order Reviews Data Set Cleaning

import pandas as pd
import numpy as np


df = pd.read_csv('olist_order_reviews_dataset.csv')

print(df.isnull().sum())
print(df.dtypes)

# Review ID has been repeated multiple times.
# Rows with duplicate review ID are identical except for order ID

print(len(df['review_id'].unique()))
repeated_rows = df[df['review_id'].duplicated(keep=False)]

print(repeated_rows)


print(len(df['order_id'].unique()))

# Changing review to a categorical feature
df['review_score'] = df['review_score'].astype('category')
print(df.dtypes)

# Create a new column for the length of characters in comment and title
df['comment_length'] = df['review_comment_message'].fillna('').str.len()
df['title_length'] = df['review_comment_title'].fillna('').str.len()

# Creating boolean columns for if there is a review comment or review comment title
df['has_review_comment_message'] = np.where(
    (df['review_comment_message'].isna()) | (df['comment_length'] < 4), 
    0, 
    1
)

df['has_review_comment_title'] = np.where(
    (df['review_comment_title'].isna()) | (df['title_length'] < 4), 
    0, 
    1
)

# Converting data columcs to date types

df['review_creation_date'] = pd.to_datetime(df['review_creation_date'], format='%d/%m/%Y %H:%M')
df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'], format='%d/%m/%Y %H:%M')

print(df.dtypes)

df['time_to_review'] = df['review_answer_timestamp'] - df['review_creation_date'] 

columns_to_drop = [
    'review_creation_date', 
    'review_answer_timestamp', 
    'review_comment_message', 
    'review_comment_title'
]
df = df.drop(columns=columns_to_drop)

# Write to CSV
df.to_csv('order_reviews_cleaned.csv', index=False)

###########################################################################################

# Order Items and products dataset Cleaning

import pandas as pd

# Read CSV files
order_item = pd.read_csv('olist_order_items_dataset.csv')
product = pd.read_csv('olist_products_dataset.csv')

# Grouping and summarizing order_item data
result = order_item.groupby(['order_id', 'product_id', 'seller_id']).agg(
product_cnt=('price', 'size'),
total_price=('price', 'sum'),
unit_price=('price', lambda x: x.sum() / x.size)
).reset_index()

# Getting each product_id's max and min price in history
product_price = order_item.groupby('product_id')['price'].agg(
max_price='max',
min_price='min'
).reset_index()

# Joining the two tables
oi = result.merge(product_price, on='product_id', how='inner')

# Exporting the oi table to a CSV file
oi.to_csv('oi.csv', index=False)

# Checking distinct counts of order_id and product_id
order_item_count = order_item['order_id'].nunique()
new_oi_count = oi['order_id'].nunique()
oi_product_count = oi['product_id'].nunique()
product_count = product['product_id'].nunique()
print(order_item_count, new_oi_count, oi_product_count, product_count)

# Checking if each product_id has more than one product_category_name
category_check = product.groupby('product_id')['product_category_name'].nunique().reset_index()
category_issue = category_check[category_check['product_category_name'] > 1]
print(category_issue) # Should return an empty dataframe

# Cleaning the product table
product_clean = product.assign(
product_vol=product['product_length_cm'] * product['product_height_cm'] * product['product_width_cm']
)[['product_id', 'product_category_name', 'product_vol', 'product_weight_g']]

# Joining order_item and product tables
order_item_join_product = oi.merge(product_clean, on='product_id', how='inner')

# Exporting the joined table to a CSV file
order_item_join_product.to_csv('groupby_order_item_join_product.csv', index=False)


####################################################################################################################################################################################

# Merging and Cleaning all the Clean Datasets

import pandas as pd

customers = pd.read_csv('customers_clean.csv')
items_products = pd.read_csv('groupby_order_item_join_product.csv')
order_payments = pd.read_csv('order_payments_dataframe.csv')
reviews = pd.read_csv('order_reviews_cleaned.csv')
orders = pd.read_csv('orders_dataset.csv')



#Merging orders and customers to get customer unique id
customers_orders = orders.merge(customers, how = 'left', left_on = 'customer_id', right_on = 'customer_id')

# Merging reviews and customers_orders with reviews on the left
merge_1 = reviews.merge(customers_orders, how = 'left', left_on = 'order_id', right_on = 'order_id')

# Mergin order_payments to the right
merge_2 = merge_1.merge(order_payments, how = 'left', left_on = 'order_id', right_on = 'order_id')

merge_3 = merge_2.merge(items_products, how = 'left', left_on = 'order_id', right_on = 'order_id')

# Checking for missing values in the merges
print(reviews.isnull().sum())
print(merge_1.isnull().sum())
print(merge_2.isnull().sum())
print(merge_3.isnull().sum())

# Creating tables of only rows with missing values for specific merged tables
#missing_merge_3 = merge_3[merge_3.isnull().any(axis=1)]
#missing_customers_orders = customers_orders[customers_orders.isnull().any(axis=1)]
#missing_merge_2 = merge_2[merge_2.isnull().any(axis=1)]
#missing_merge_1 = merge_1[merge_1.isnull().any(axis=1)]

final_df = merge_3

# Removing all NA Values (I am assuming that these missing values exist because of mapping errors (for exmaple some order ids dont have any review ids causing na rows))
final_df = final_df.dropna()

# Converting time to review from object dtype to float64 (seconds till review)
final_df['time_to_review'] = pd.to_timedelta(final_df['time_to_review']).dt.total_seconds()
# Removing duplicated review ids where time to review is minimum
final_df = final_df.loc[final_df.groupby("review_id")["time_to_review"].idxmax()]


print(len(final_df['review_id'].unique()))
print(len(final_df['order_id'].unique()))

final_df['review_id'] = final_df['review_id'].astype(str)
final_df['order_id'] = final_df['order_id'].astype(str)

# find orders with more than 1 review_id
multi_review_orders = final_df.groupby('order_id')['review_id'].nunique()

# how many order_id has different review_id: 250
len(multi_review_orders[multi_review_orders > 1]) # 250
# find these order_id
multi_review_index = multi_review_orders[multi_review_orders > 1].index
# remove all order_id with more than 1 review_id
final_df= final_df[~final_df['order_id'].isin(multi_review_index)]

# check number of different order_id and review_id, found out they are not equal
final_df['order_id'].nunique()
final_df['review_id'].nunique()

# Removing rows where product weight is 0 and 2
final_df = final_df[final_df['product_weight_g'] > 2]

# Removing rows where payment type is not defined
final_df = final_df[final_df['payment_type_not_defined'] != 1]

# Checking to see if there are any rows where payment_value < total_price
final_df[final_df['payment_value'] < final_df['total_price']]
# remove all order_id where payment_value < total_price
final_df = final_df[final_df['payment_value'] >= final_df['total_price']]

# Columns to be dropped
drop_columns = ["payment_type_not_defined", "min_price", 'max_price', 'total_price']
final_df = final_df.drop(columns=drop_columns)

# Using a customers average reviews to create a new class @Nadim

# Step 1: Aggregate review scores for each customer 
customer_review_stats = final_df.groupby('customer_unique_id')['review_score'].agg(['mean', 'std']).reset_index() 
customer_review_stats.columns = ['customer_unique_id', 'mean_review_score', 'std_review_score'] 
# Fill NaN for std_review_score (e.g., customers with only one review have NaN std) 
customer_review_stats['std_review_score'] = customer_review_stats['std_review_score'].fillna(0) 


# Step 2: Define thresholds for classification 
def classify_customer(row): 

    if row['mean_review_score'] >= 4.0 and row['std_review_score'] <= 1.0: 

        return 'Good' 

    elif 2.5 <= row['mean_review_score'] < 4.0: 

        return 'Mediocre' 

    else: 

        return 'Bad' 

 
# Apply classification logic 
customer_review_stats['reviewer_type'] = customer_review_stats.apply(classify_customer, axis=1) 
# Step 3: Merge back to the main dataset 
final_df = final_df.merge(customer_review_stats[['customer_unique_id', 'reviewer_type']], on='customer_unique_id', how='left') 

drop_columns = ["review_id", 'order_id', 'comment_length','title_length','has_review_comment_message','has_review_comment_title','time_to_review','customer_id','order_status','customer_unique_id','customer_zip_code_prefix','product_id','seller_id','reviewer_type']
final_df = final_df.drop(columns=drop_columns)

final_df.to_csv('df.csv', index=False)

#####################################################################################################################################################################







