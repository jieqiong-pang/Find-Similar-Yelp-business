# Find-Similar-Yelp-business
using MinHash and Locality Sensitive Hashing algorithms with Jaccard similarity(>=0.05) to find similar business pairs in the train_review.json file. The accuracy >= 0.8; execution time < 200 sec

## steps:
1. transfer the actual ratings/stars in the reviews to 0 or 1 ratings (if a user has rated a business, the user's contribution in the characteristic matrix is 1, if the user hasn't rated the business, the contribution is 0), and built business-user characteristic matrix
2. select hash function to do permutation in characteristic matrix
3. built the signature matrix using Min-Hash
4. use LSH more efficient at finding business that might be similar, divide the matrix into b bands with r rows(set b and r properly to balance the number of candidates and computational cost)
5. two businesses become a candidate pair if their signattures are identical in at least one band
6. verify the candidate pairs using their original Jaccard similarity

## exectuion commend
$ spark-submit task1.py <input_file> <output_file>

