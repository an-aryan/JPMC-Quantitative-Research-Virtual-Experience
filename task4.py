import numpy as np
import pandas as pd
def log_likelihood(n, k):
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)


def quantize_fico_scores(fico_scores, num_buckets):
    min_score = min(fico_scores)
    max_score = max(fico_scores)
    bucket_width = (max_score - min_score) / num_buckets
    
    rating_map = {}
    for score in fico_scores:
        bucket_index = int((score - min_score) / bucket_width)
        rating_map[score] = bucket_index
    
    return rating_map

# Read loan data from CSV
df = pd.read_csv('JPMC\Task 3 and 4_Loan_Data.csv')

x = df['default'].tolist()
y = df['fico_score'].tolist()

# Quantize FICO scores into rating buckets
rating_map = quantize_fico_scores(y, num_buckets=5)

# Calculate default and total counts for each rating bucket
default = [0 for _ in range(5)]
total = [0 for _ in range(5)]

# print(rating_map[y[2659]])

for i in range(len(x)-1):
    fico_score = y[i]
    rating = rating_map[fico_score] -1
    # print(default[rating], i)
    default[rating] += x[i]
    total[rating] += 1

for i in range(1, 5):
    default[i] += default[i-1]
    total[i] += total[i-1]

# Log-likelihood calculation and dynamic programming
r = 10
dp = [[[-float('inf'), 0] for _ in range(5)] for _ in range(r+1)]

for i in range(r+1):
    for j in range(5):
        if i == 0:
            dp[i][j][0] = 0
        else:
            for k in range(j):
                if total[j] == total[k]:
                    continue
                if i == 1:
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    likelihood = log_likelihood(total[j]-total[k], default[j]-default[k])
                    if dp[i][j][0] < (dp[i-1][k][0] + likelihood):
                        dp[i][j][0] = likelihood + dp[i-1][k][0]
                        dp[i][j][1] = k

# Retrieve the FICO score buckets for the best rating map
k = 4
rating_buckets = []
while r >= 0:
    rating_buckets.append(k)
    k = dp[r][k][1]
    r -= 1

# Map the rating buckets to their respective FICO score ranges
min_score = min(y)
max_score = max(y)
bucket_width = (max_score - min_score) / 5

rating_ranges = []
for bucket in rating_buckets[::-1]:
    start_range = round(min_score + (bucket_width * bucket))
    end_range = round(min_score + (bucket_width * (bucket + 1)))
    rating_ranges.append((start_range, end_range))

print(rating_ranges)
