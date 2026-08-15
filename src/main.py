from data_loader import load_dataset
from pathlib import Path
import matplotlib.pyplot as plt


FEATURE_NAMES = ("size_sqft", "bedrooms", "floors", "age_years")
DATA_DIR = Path(__file__).resolve().parent / "data"

def load_house_data(filename: str = "houses.txt"):
    return load_dataset(DATA_DIR / filename, expected_columns=len(FEATURE_NAMES) + 1)

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()