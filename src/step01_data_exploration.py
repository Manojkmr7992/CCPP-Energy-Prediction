from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "CCPP_data.csv"

# 1 & 2. Load dataset without modifying or cleaning
df = pd.read_csv(DATA_PATH)

# 3. First 10 rows
print("=== First 10 Rows ===")
print(df.head(10).to_string())
'''
=== First 10 Rows ===
      AT      V       AP     RH      PE
0  14.96  41.76  1024.07  73.17  463.26
1  25.18  62.96  1020.04  59.08  444.37
2   5.11  39.40  1012.16  92.14  488.56
3  20.86  57.32  1010.24  76.64  446.48
4  10.82  37.50  1009.23  96.62  473.90
5  26.27  59.44  1012.23  58.77  443.67
6  15.89  43.96  1014.02  75.24  467.35
7   9.48  44.71  1019.12  66.43  478.42
8  14.64  45.00  1021.78  41.25  475.98
9  11.74  43.56  1015.14  70.72  477.50
'''

# 4. Last 5 rows
print("\n=== Last 5 Rows ===")
print(df.tail(5).to_string())
'''
=== Last 5 Rows ===
         AT      V       AP     RH      PE
9563  16.65  49.69  1014.01  91.00  460.03
9564  13.19  39.18  1023.67  66.78  469.62
9565  31.32  74.33  1012.92  36.48  429.57
9566  24.48  69.45  1013.86  62.39  435.74
9567  21.60  62.52  1017.23  67.87  453.28
'''

# 5. Total rows and columns
rows, cols = df.shape
print(f"\n=== Dataset Dimensions ===\nRows: {rows:,} | Columns: {cols}")

'''
=== Dataset Dimensions ===
Rows: 9,568 | Columns: 5
'''
# 6. Column names and data types
print("\n=== Column Names & Data Types ===")
print(df.dtypes)

'''
=== Column Names & Data Types ===
AT    float64
V     float64
AP    float64
RH    float64
PE    float64
dtype: object
'''
# 7. Missing values per column
print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

'''
=== Missing Values per Column ===
AT    0
V     0
AP    0
RH    0
PE    0
dtype: int64
'''

# 8. Exact duplicate rows
print(f"\n=== Duplicate Rows ===\nTotal duplicates: {df.duplicated().sum():,}")

''' 
=== Duplicate Rows ===
Total duplicates: 41
''' 

# 9. Descriptive statistics
print("\n=== Descriptive Statistics ===")
print(df.describe().round(4).to_string())
'''
=== Descriptive Statistics ===
              AT          V         AP         RH        PE
count  9568.0000  9568.0000  9568.0000  9568.0000  9568.000
mean     19.6512    54.3058  1013.2591    73.3090   454.365
std       7.4525    12.7079     5.9388    14.6003    17.067
min       1.8100    25.3600   992.8900    25.5600   420.260
25%      13.5100    41.7400  1009.1000    63.3275   439.750
50%      20.3450    52.0800  1012.9400    74.9750   451.550
75%      25.7200    66.5400  1017.2600    84.8300   468.430
max      37.1100    81.5600  1033.3000   100.1600   495.760
'''