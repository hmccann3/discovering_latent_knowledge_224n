from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS
import numpy as np

from sklearn.calibration import calibration_curve
import matplotlib.lines as line
import matplotlib.pyplot as plt


from sklearn.calibration import CalibratedClassifierCV as cccv

def main(args, generation_args):
    # load hidden states and labels
    name = f"{generation_args.dataset_name}_on_{args.calibration_dataset_name}_{args.calibration_type}.png"
    neg_hs, pos_hs, y = load_all_generations(generation_args)

    if args.calibration_dataset_name is not None:
        generation_args.dataset_name = args.calibration_dataset_name
    cali_neg_hs, cali_pos_hs, cali_y = load_all_generations(generation_args)
    cali_neg_hs, cali_pos_hs = cali_neg_hs[..., -1], cali_pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        cali_neg_hs = cali_neg_hs.squeeze(1)
        cali_pos_hs = cali_pos_hs.squeeze(1)

    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
    pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
    y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

    cali_neg_hs_train, cali_neg_hs_test = cali_neg_hs[:len(cali_neg_hs) // 2], cali_neg_hs[len(cali_neg_hs) // 2:]
    cali_pos_hs_train, cali_pos_hs_test = cali_pos_hs[:len(cali_pos_hs) // 2], cali_pos_hs[len(cali_pos_hs) // 2:]
    cali_y_train, cali_y_test = cali_y[:len(cali_y) // 2], cali_y[len(cali_y) // 2:]

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size,
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay,
                    var_normalize=args.var_normalize)

    fitted_ccs = ccs.fit(None, None)
    #ccs.repeated_train()

    ccs_acc, y_preds_pos, y_preds_neg = ccs.get_acc(neg_hs_test, pos_hs_test, y_test, return_conf=True)
    uncali_x, uncali_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
    if uncali_x[0] > uncali_y[-1] or uncali_y[0] > uncali_y[-1]:
        print('Flipping uncalibrated graph')
        uncali_x, uncali_y = calibration_curve(y_test, y_preds_neg.flatten(), n_bins=10)
    # Assume that the first point should be less than the last point when plotting
    print(f"CCS uncalibrated accuracy: {ccs_acc}")

    X = np.stack([cali_pos_hs_train, cali_neg_hs_train], axis=0).transpose(1, 0, 2)
    y = cali_y_train

    #Calibration using BCE
    if args.calibration_type == 'bce':
        calibrated_ccs = fitted_ccs.fit(X, y)
        ccs_acc, y_preds_pos, y_preds_neg = calibrated_ccs.get_acc(neg_hs_test, pos_hs_test, y_test, return_conf=True)
        cali_x, cali_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
        if cali_x[0] > cali_x[-1] or cali_y[0] > cali_y[-1]:
            print('Flipping calibrated graph')
            cali_x, cali_y = calibration_curve(y_test, y_preds_neg.flatten(), n_bins=10)

    # Calibration with sklearn
    elif args.calibration_type in ['sigmoid', 'isotonic']:
        calibrated_ccs = cccv(fitted_ccs, cv='prefit', method=args.calibration_type)
        calibrated_ccs.fit(X, y)
        X = np.stack([pos_hs_test, neg_hs_test], axis=0).transpose(1, 0, 2)
        y_preds = calibrated_ccs.predict_proba(X)
        predictions = (y_preds[:,1] < 0.5).astype(int)
        acc = (predictions == y_test).mean()
        ccs_acc = max(acc, 1 - acc)
        cali_x, cali_y = calibration_curve(y_test, y_preds[:,1].flatten(), n_bins=10)
        if cali_x[0] > cali_x[-1] or cali_y[0] > cali_y[-1]:
            print('Flipping calibrated graph')
            cali_x, cali_y = calibration_curve(y_test, y_preds[:,0].flatten(), n_bins=10)
    print(f"CCS calibrated accuracy: {ccs_acc}")


    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    plt.plot(uncali_y, uncali_x, marker = '.', label = 'CCS')
    plt.plot(cali_y, cali_x, marker = 'x', label = 'Calibrated CCS')
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.savefig(name)



if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    cdn = generation_args.calibration_dataset_name
    del generation_args.calibration_dataset_name
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    parser.add_argument("--calibration_type", type=str, default="bce")
    args = parser.parse_args()
    args.calibration_dataset_name = cdn
    main(args, generation_args)
