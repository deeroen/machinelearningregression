from data_treatments import *
import matplotlib.pyplot as plt

def visualize(X, Y, feature_names, indexs):
    l = len(indexs)
    if l < 4:
        for i in indexs:
            fig = plt.figure(4, figsize=(10, 10), dpi=90)
            plt.scatter(X[:, i], Y, s=10)
            plt.title(f"{feature_names[i]}")
            plt.tight_layout()
            plt.show()

    else:
        fig = plt.figure(4, figsize=(10, 10), dpi=90)
        for i in range(l):
            plt.subplot(9 // 3 + 1, 3, i + 1)
            plt.scatter(X[:, indexs[i]], Y, s=10)
            plt.title(f"{feature_names[indexs[i]]}")
        plt.tight_layout()
        plt.show()
    return 0


# visualization of raw data
n_samples_X1, n_feats_X1 = X1.shape
n_samples_X2, n_feats_X2 = X2.shape
feature_names = ["year", "month", "day", "hour", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station", "PM2.5"]
X1 = X1.to_numpy()
Y1 = Y1.to_numpy()

#fig=plt.figure(figsize=(10, 10), dpi=90)
#for i in range(0,15):
#    plt.subplot(n_feats_X1 // 3 + 1, 3, i + 1)
#    plt.scatter(X1[:, i], Y1, s=10)
#    plt.title(f"{feature_names[i]}")
#plt.tight_layout()
#plt.show()


# visualization of data after data_treatment

X1_treated = worktbl.to_numpy()
n_samples_X1_treated, n_feats_X1_treated = X1_treated.shape
feature_names_treated = list(worktbl)

fig=plt.figure(figsize=(10, 10), dpi=90)
for i in range(9):
    plt.subplot(9 // 3 + 1, 3, i + 1)
    plt.scatter(X1_treated[:, i], Y1, s=10)
    plt.title(f"{feature_names_treated[i]}")
plt.tight_layout()
plt.show()


fig=plt.figure(2, figsize=(10, 10), dpi=90)
for i in range(9):
    plt.subplot(9 // 3 + 1, 3, i + 1)
    plt.scatter(X1_treated[:, i+9], Y1, s=10)
    plt.title(f"{feature_names_treated[i+9]}")
plt.tight_layout()
plt.show()


fig=plt.figure(3, figsize=(10, 10), dpi=90)
for i in range(9):
    plt.subplot(9 // 3 + 1, 3, i + 1)
    plt.scatter(X1_treated[:, i+18], Y1, s=10)
    plt.title(f"{feature_names_treated[i+18]}")
plt.tight_layout()
plt.show()

fig=plt.figure(4, figsize=(10, 10), dpi=90)
for i in range(6):
    plt.subplot(9 // 3 + 1, 3, i + 1)
    plt.scatter(X1_treated[:, i+27], Y1, s=10)
    plt.title(f"{feature_names_treated[i+27]}")
plt.tight_layout()
plt.show()
