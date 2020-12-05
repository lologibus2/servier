import copy
import matplotlib.pyplot as plt


def plot_confusion_wiki(confmat):
    fig, ax = plt.subplots(figsize=(5, 5))
    confmat_wiki = copy.copy(confmat.T)
    confmat_wiki[0][0] = confmat[1][1]
    confmat_wiki[1][1] = confmat[0][0]
    ax.matshow(confmat_wiki, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat_wiki.shape[0]):
        for j in range(confmat_wiki.shape[1]):
            ax.text(x=j, y=i, s=confmat_wiki[i, j], va='center', ha='center')
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.tight_layout()
