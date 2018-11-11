from sklearn.metrics import confusion_matrix, classification_report
from share.global_setting import ACTIONS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


fold = 0
cms = []
cm_sum = np.zeros((7, 7), dtype='int')
y_test_all = []
pred_all = []
for fold in range(5):
    test_result = pd.read_csv(r"./data/test/fold{0}.csv".format(fold))
    y_test = test_result['gt'].tolist()
    pred = test_result['predict'].tolist()

    y_test_all.extend(y_test)
    pred_all.extend(pred)

    cm = confusion_matrix(y_test, pred)
    cm_sum += cm

    cms.append(cm)


print(classification_report(y_test_all, pred_all))
plot_confusion_matrix(cm_sum, ACTIONS)
