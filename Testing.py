import os
import re
import yaml
import torch
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from models.data_loader import build_data_loader
from models.lstm import build_lstm
from Evaluation import calculate_precision, calculate_recall, calculate_f1_score

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"


def calculate_binary_classifier(gt, hypo):
    TP, FP, FN = 0, 0, 0
    for idx, x in enumerate(gt):
        #print(x, hypo[idx])
        if x == 1 and round(hypo[idx]) == 1:
            TP += 1
        elif x == 0 and round(hypo[idx]) == 1:
            FP += 1
        elif x == 1 and round(hypo[idx]) == 0:
            FN += 1
    return TP, FP, FN


@torch.no_grad()
def test():
    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_dir = config['training']['saved_dir']
    model_file = config['training']['saved_best_model'].join(
                ("best",
                 config['model']['name'] + "_" + str(config['model']['hidden_dim1']) + "_" + str(config['model']['hidden_dim2']),
                 config['data']['pkl_name'].split('.')[-2].split('-')[-1],
                 "window" + str(config['model']['seq_length_s']) + "sec",
                 config['optimizer']['name'],
                 "batch" + str(config['training']['batch_size']) + ".pth"
                 )
            )

    model = torch.load(os.path.join(model_dir, model_file))

    plot_filename = re.sub('best', 'results', model_file)

    model.eval()

    # Load test data_loader
    test_loader = build_data_loader(config, 'test')

    gts = []
    preds = []
    elapsed_time = []

    # Shape of inputs: (batch_size, sequence_length, num_features)
    # Shape of targets: (batch_size, sequence_length, binary_label)
    for idx, (inputs, targets) in enumerate(test_loader):
        start_time = time.time()

        pred = model(inputs)

        elapsed_time.append(time.time() - start_time)
        #print("Out shape: ", pred.shape)
        #print(pred, targets)
        preds.append(float(torch.sigmoid(pred[0][-1][0]).detach()))

        gts.append(float(targets[0][-1][0].detach()))

    print("Inference mean latency = %.02f ms" % (sum(elapsed_time)/len(elapsed_time) * 1000))

    ### METRICS ###
    print(gts[:20], preds[:20])
    # aggregate
    tp, fp, fn = calculate_binary_classifier(gts, preds)
    print("TP = {}, FP = {}, FN = {}, among a total of {} testing samples.".format(tp, fp, fn, len(test_loader)))
    print()
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    F1 = calculate_f1_score(precision, recall)

    print("Precision = {}, Recall = {}, F1 score = {}".format(precision, recall, F1))

    FPr, TPr, thresholds = roc_curve(gts, preds, sample_weight=None, drop_intermediate=True)

    # Calculate the AUC score
    roc_auc = auc(FPr, TPr)
    print("AUC = {}".format(roc_auc))

    # Calculate precision and recall
    Precision, Recall, _ = precision_recall_curve(gts, preds)

    # Calculate the area under the precision-recall curve (AUC-PR)
    auc_pr = average_precision_score(gts, preds)
    print("Precision-Recall Curve = {}".format(auc_pr))

    # Plotting the ROC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    font = {'size': 18}
    matplotlib.rc('font', **font)

    ax1.plot(FPr, TPr, color='orange', lw=3, label='AUC = %0.3f' % roc_auc)
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=19)
    ax1.set_ylabel('True Positive Rate', fontsize=19)
    ax1.set_title('ROC Curve', fontsize=21)
    ax1.legend(loc='lower right', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=15)

    # Plotting the Precision-Recall curve
    ax2.plot(Recall, Precision, color='green', lw=3, label='AUC-PR = %0.3f' % auc_pr)
    ax2.set_xlabel('Recall', fontsize=19)
    ax2.set_ylabel('Precision', fontsize=19)
    ax2.set_title('Precision-Recall Curve', fontsize=21)
    ax2.legend(loc='upper right', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='minor', labelsize=15)

    #fig.suptitle('Precision %.3f; Recall %.3f; F1 %.3f' % (precision, recall, F1), fontsize=22)

    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)

    plt.tight_layout()
    plt.savefig(os.path.join('./imgs', '{}.pdf'.format(plot_filename[:-4])), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    test()
