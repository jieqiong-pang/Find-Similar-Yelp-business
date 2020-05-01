from pyspark import SparkContext, SparkConf, StorageLevel
import sys
import json
import itertools
import time
import random

start = time.time()

minhash_number = 40
band = 40
row = int(minhash_number / band)
random.seed(9)
a_l = [random.randint(0, (2 ** 32) - 1) for i in range(minhash_number)]
b_l = [random.randint(0, (2 ** 32) - 1) for i in range(minhash_number)]
def minhash(users_number, users_count, minhash_number):
    global a_l
    global b_l

    signatures = []
    for i in range(minhash_number):
        min_value = float("inf")
        for user_number in users_number:
            value = (a_l[i] * user_number + b_l[i]) % users_count
            min_value = min(min_value, value)
        signatures.append(min_value)
    return signatures


def LSH(b_sig, row):
    business_id = b_sig[0]
    signatures = b_sig[1]
    # 'i' is band number
    ls = []
    for i in range(0, minhash_number, row):
        band_signatures = signatures[i: row + i]
        bucket_number = hash(tuple(band_signatures))
        ls.append(((i, bucket_number), business_id))
    return ls


input_file = sys.argv[1]
output_file = sys.argv[2]
conf = (
    SparkConf()
        .setAppName("inf553_hw3_task1")
        .set("spark.executor.memory", "4g")
        .set("spark.driver.memory", "4g")
)
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

rdd = (
    sc.textFile(input_file)
        .map(json.loads)
        .persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
)

# give number to user_id/business_id from 0 to last user_id/business_id
users_number = (
    rdd.map(lambda user: user["user_id"]).distinct().zipWithIndex().collectAsMap()
)
users_count = len(users_number)

business_number = (
    rdd.map(lambda x: x["business_id"]).distinct().zipWithIndex().collectAsMap()
)
# ====>{number:business_id,number:business_id...}
change = {value: key for key, value in business_number.items()}

# [(business_number,{user_number...}),(business_number,{user_number...})...]
# RDD[bid, [uid0, uid1, ...]]
business = (
    rdd.map(
        lambda business_users: (
            business_number[business_users["business_id"]],
            users_number[business_users["user_id"]],
        )
    )
        .groupByKey()
        .mapValues(set)
        .persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
)

# {business_number:{user_number,user_number...},business_number:{user_number,user_number...}...}
business_users = business.collectAsMap()
del users_number

# signature matrix [(businsee_id,[signatures]),(businsee_id,[signatures]),(businsee_id,[signatures])]
business_signatures = business.mapValues(
    lambda users_number: minhash(users_number, users_count, minhash_number)
)

# [(b1,b2,sim),(b1,b2,sim)....]
lsh = (
    business_signatures.flatMap(lambda b_sig: LSH(b_sig, row))
        .groupByKey()
        .mapValues(list)
        .filter(lambda l: len(l[1]) >= 2)  # [[business_id,business_id,business_id...],[business_id,business_id,business_id...],[business_id,business_id,business_id...]]
        .flatMap(
        lambda b: set(itertools.combinations(sorted(b[1]), 2))
    )  # [(pair),(pair)...]
        .collect()
)

with open(output_file, "w") as f:
    true_positive = 0
    candidate_set = set(lsh)
    for candidate in candidate_set:
        business1 = candidate[0]
        business2 = candidate[1]
        intersection = business_users[business1].intersection(business_users[business2])
        union = business_users[business1].union(business_users[business2])
        similarity = len(intersection) / len(union)
        if similarity >= 0.05:
            true_positive += 1
            output = {
                "b1": change[business1],
                "b2": change[business2],
                "sim": similarity,
            }
            f.write(json.dumps(output) + "\n")
end = time.time()
print("Duration: " + str(end - start))