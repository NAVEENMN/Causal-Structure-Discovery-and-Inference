import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv\
    (os.path.join(os.getcwd(), 'data', 'simulation.csv'))
print(data.head())
sns.scatterplot(data.s_p0_p1, data.v_p0_xdim)
plt.show()