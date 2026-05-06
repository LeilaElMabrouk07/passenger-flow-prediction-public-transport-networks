
import pandas as pd
import numpy as np
from itertools import combinations
import sklearn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

df = pd.read_csv("london_fri_16_24.csv")

df_train  = df[ (df['year'] != 24) &
                (df['Line'].isin(['Bakerloo', "H&C and Circle"]) &
                (df['Dir'] =="NB" ))
]
df_test = df[(df['year'] == 24) &
 (df['Line'].isin(['Bakerloo', "H&C and Circle"])) &
            ( df['Dir'] == "NB")]

columns_list = df_train.columns
df_test, df_train.shape

From_station, To_Station = df_test['From Station'].tolist(),df_test['To Station'].tolist()


import numpy as np
import pandas as pd

From_station = df_test['From Station'].tolist()
To_Station   = df_test['To Station'].tolist()

index_beg = 18
index_end = -1

years = list(range(16, 24))

inform_col = columns_list[:index_beg]
values_col = columns_list[index_beg:index_end]

# Initialisation de la matrice d'agrégation
df_agg = np.zeros((24, len(values_col)))

for y in years:
    df_y = df_train[df_train['year'] == y].reset_index(drop=True)

    df_v = df_y[values_col]

    print('year', y, df_v.shape)

    df_agg += df_v.to_numpy()

# Moyenne sur les années
df_agg = df_agg / len(years)

# Transformer en DataFrame
df_agg_df = pd.DataFrame(df_agg, columns=values_col)

# Partie info : on prend celle de df_test
df_info = df_test[inform_col].reset_index(drop=True).copy()

# Reconstruction finale
df_mean = pd.concat([df_info, df_agg_df], axis=1)

print(df_mean.shape)
df_mean.head()

print( df_test.shape)

df_test_val = df_test[values_col]
df_test_val = df_test_val.to_numpy()


df_diff = df_test_val - df_agg
df_diff = np.abs(df_diff)
print(df_diff.mean() )

idx_time = 10

X = [fr + " → " + to for fr, to in zip(From_station, To_Station)]

Y_1 = df_test_val[:, idx_time]
Y_2 = df_agg[:, idx_time]

MAE = np.float16((np.abs(Y_1 - Y_2)).mean())

plt.figure(figsize=(16,6))

plt.plot(X, Y_1, marker='o', label='Observed 2024')
plt.plot(X, Y_2, marker='s', label='Mean 2016-2023')

plt.ylabel("Value")
plt.title(f"Comparison between observed values (2024) and historical mean | MAE {MAE}")

plt.xticks(rotation=60, ha="right")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.show()