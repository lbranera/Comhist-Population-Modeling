import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools import exponential, logistic
from tools import get_exponential_params, get_logistic_params

df = pd.read_csv("data.csv")
df["t_year"] = df["year"] - df.iloc[0]["year"]

r, P_init = get_exponential_params(df["t_year"], df["population"])

years = [i for i in range(1850, 2021)]

predicted_populations = [exponential(year - df.iloc[0]["year"], r, P_init) for year in years]
plt.plot(years, predicted_populations, color="blue", label="exponential")

K, r, P_init = get_logistic_params(df["t_year"], df["population"])

predicted_populations = [logistic(year - df.iloc[0]["year"], K, r, P_init) for year in years]
plt.plot(years, predicted_populations, color="green", label="logistic")

plt.scatter(df["year"], df["population"], color="red", label="Census Data")

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Over Time")
plt.legend()
plt.show()
