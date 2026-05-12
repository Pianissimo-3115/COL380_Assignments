import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("speedup.csv")

plt.figure()
plt.plot(df["threads"], df["updateDisplay_speedup"], marker="o", label="updateDisplay")
plt.plot(df["threads"], df["printOrderStats_speedup"], marker="o", label="printOrderStats")
plt.plot(df["threads"], df["totalAmountTraded_speedup"], marker="o", label="totalAmountTraded")

plt.xlabel("Number of threads")
plt.ylabel("Speedup (Sequential / Parallel)")
plt.title("OpenMP Speedup vs Number of Threads")
plt.legend()
plt.grid(True)

plt.savefig("speedup.png")
plt.show()
