import argparse
import pickle
from utility import make_dir
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(output_dir, confusion_plot_data):
    cols = 2
    rows = len(confusion_plot_data)

    fig, axes = plt.subplots(rows, cols, figsize = (10 * cols, 10 * rows))
    axes = axes.flatten()

    axis_idx = 0
    for i, (confusion_matrices, label_classes) in enumerate(confusion_plot_data):
        label_classes = [labels for labels in label_classes if labels]


        for cm, label_class in zip(confusion_matrices, label_classes):
            ax = axes[axis_idx]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = label_class)
            disp.plot(ax=ax, xticks_rotation="vertical")

            ax.set_title(f"Experiment {i} - {'Par Boundaries' if 'par_start' in label_class else 'Ch Boundaries'}")
            axis_idx += 1

    for ax in axes[axis_idx]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_confusion_matrices.png"))
    plt.close(fig)

def write_results(output_dir, experiment_results):

    with open(os.path.join(output_dir, "final_results.txt"), "wt") as output_file:
        for i, (exp_acc, exp_f1, class_labels) in enumerate(experiment_results):
            output_file.write(f"Experiment {i} || {list(zip(['Par Accuracies', 'Ch Accuracies'], exp_acc))} || {list(zip(class_labels, exp_f1))}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest = "data", nargs = "*", help = "Train, dev, and test file paths")
    parser.add_argument("--target", dest = "target", help = "Final report directory")
    args, rest = parser.parse_known_args()
    
    make_dir(args.target)

    confusion_plot_data = []
    exp_results = []
    for result_file in args.data:
        with open(result_file, "rb") as fp:
            (train_losses, dev_losses, cms, classes, accuracies, f1s) = pickle.load(fp)
            confusion_plot_data.append((cms, classes))
            exp_results.append((accuracies, f1s, classes))
            print("Document found")

    # plot_confusion_matrix(args.target, confusion_plot_data)
    write_results(args.target, exp_results)