import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

data = pd.read_csv("project_train.csv")

# data.hist(bins=30)
# plt.show()


rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

# corr = data.corr()
# corr.style.background_gradient(cmap='coolwarm')

# plt.matshow(corr)
# plt.show()