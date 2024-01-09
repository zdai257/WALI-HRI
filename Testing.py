import os
import yaml
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from models.data_loader import build_data_loader
from models.lstm import build_lstm
from Evaluation import *


@torch.no_grad()
def test():
    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_dir = config['training']['saved_dir']
    model_file = config['training']['saved_best_model']

    model = torch.load(os.path.join(model_dir, model_file))

    model.eval()

    # Load test data_loader
    test_loader = build_data_loader(config, 'test')

    gts = []
    preds = []
    elapsed_time = []

    # Shape of inputs: (batch_size, sequence_length, num_features)
    # Shape of targets: (batch_size, sequence_length, binary_label)
    for idx, (inputs, targets) in enumerate(test_loader):

        pred = model(inputs)

        #print("Out shape: ", pred.shape)
        #print(pred, targets)
        preds.append(float(torch.sigmoid(pred[0][0]).detach()))

        gts.append(float(targets[0][0].detach()))

    print(gts[:100], preds[:100])
    # TODO fix
    tp, fp, fn = calculate_metrics(preds, gts)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    F1 = calculate_f1_score(precision, recall)
    print(tp, fp)
    print(tp, fn)
    print("Precision = {}, Recall = {}, F1 score = {}".format(precision, recall, F1))

    FPr, TPr, thresholds = roc_curve(gts, preds, sample_weight=None, drop_intermediate=True)

    # Calculate the AUC score
    roc_auc = auc(FPr, TPr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPr, TPr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join('./imgs', 'roc_curve.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    test()
