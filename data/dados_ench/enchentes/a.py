import pandas as pd

df = pd.read_csv ("01_06_2022.csv")
df.drop(columns=["humidity_average"], inplace = True)
print(df.columns)
df.to_csv("01_06_2022.csv")