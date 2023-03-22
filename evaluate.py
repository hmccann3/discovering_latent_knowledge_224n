from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS
import numpy as np
import pickle
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.lines as line
import matplotlib.pyplot as plt


from sklearn.calibration import CalibratedClassifierCV as cccv

def main(args, generation_args):
    # load hidden states and labels
    if not os.path.exists("imgs_imdb"):
        os.mkdir("imgs_imdb")
    if not os.path.exists("scores_imdb"):
        os.mkdir("scores_imdb")

    prompt_idx = generation_args.prompt_idx
    name = f"imgs_imdb/{generation_args.num_examples}sample_prompt{generation_args.prompt_idx}_{generation_args.dataset_name}"
    #name = f"{generation_args.num_examples}sample_prompt{generation_args.prompt_idx}_{generation_args.dataset_name}"
    if args.use_dropout and args.use_dropout_loss:
        name += "_drop_and_droploss.png"
    elif args.use_dropout:
        name += "_drop.png"
    elif args.use_dropout_loss:
        name += "_droploss.png"
    else:
        name += f"_{args.calibration_type}.png"
    neg_hs, pos_hs, y = load_all_generations(generation_args)

    if args.calibration_dataset_name is not None:
        generation_args.dataset_name = args.calibration_dataset_name
    cali_neg_hs, cali_pos_hs, cali_y = load_all_generations(generation_args)
    cali_neg_hs, cali_pos_hs = cali_neg_hs[..., -1], cali_pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        cali_neg_hs = cali_neg_hs.squeeze(1)
        cali_pos_hs = cali_pos_hs.squeeze(1)

    if args.eval_dataset_name is not None:
        generation_args.dataset_name = args.eval_dataset_name
        generation_args.prompt_idx = args.eval_prompt_idx
        eval_neg_hs, eval_pos_hs, eval_y = load_all_generations(generation_args)
        eval_neg_hs, eval_pos_hs = eval_neg_hs[..., -1], eval_pos_hs[..., -1]  # take the last layer
        if eval_neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
            eval_neg_hs = eval_neg_hs.squeeze(1)
            eval_pos_hs = eval_pos_hs.squeeze(1)
        _, eval_neg_hs_test = eval_neg_hs[:len(eval_neg_hs) // 2], eval_neg_hs[len(eval_neg_hs) // 2:]
        _, eval_pos_hs_test = eval_pos_hs[:len(eval_pos_hs) // 2], eval_pos_hs[len(eval_pos_hs) // 2:]
        _, eval_y_test = eval_y[:len(eval_y) // 2], eval_y[len(eval_y) // 2:]


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

    all_results = {}

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    y_preds = lr.predict_proba(x_test)
    y_preds_pos = y_preds[:,1]
    lr_acc = lr.score(x_test, y_test)
    lr_brier = brier_score_loss(y_test, y_preds_pos.flatten())
    lr_x, lr_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
    print("Logistic regression accuracy: {}".format(lr_acc))
    print(f"Brier score: {lr_brier}")
    all_results["lr_acc"] = lr_acc
    all_results["lr_brier"] = lr_brier

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size,
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay,
                    var_normalize=args.var_normalize, use_dropout=False, use_dropout_loss=False,
                    dropout_loss_weight=args.dropout_loss_weight, dropout_factor=args.dropout_factor, confidence_loss_scale=1.0)

    fitted_ccs = ccs.fit(None, None)
    #ccs.repeated_train()

    ccs_acc, y_preds_pos, y_preds_neg = ccs.get_acc(neg_hs_test, pos_hs_test, y_test, return_conf=True)
    uncali_x, uncali_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
    if (uncali_x[0] < uncali_x[-1] and uncali_y[0] > uncali_y[-1]) or (uncali_x[0] > uncali_x[-1] and uncali_y[0] < uncali_y[-1]):
        #print('Flipping uncalibrated graph')
        uncali_y = uncali_y[::-1]
        temp = y_preds_pos
        y_preds_pos = y_preds_neg
        y_preds_neg = temp
        #uncali_x, uncali_y = calibration_curve(y_test, y_preds_neg.flatten(), n_bins=10)
    # Assume that the first point should be less than the last point when plotting
    uncali_brier = brier_score_loss(y_test, y_preds_pos.flatten())
    print(f"CCS uncalibrated accuracy: {ccs_acc}")
    print(f"Brier score: {uncali_brier}")
    all_results["ccs_acc"] = ccs_acc
    all_results["ccs_brier"] = uncali_brier

    uncali_y_preds_pos = y_preds_pos.flatten()

    #plt.hist(y_preds_pos.flatten(), bins=10, range=[0, 1], edgecolor='b')
    #plt.xlabel('Probability classes (intervals of 0.1)')
    #plt.ylabel('Number of predictions per class')
    #plt.savefig("ccs_uncalibrated_histogram.png")

    X = np.stack([cali_pos_hs_train, cali_neg_hs_train], axis=0).transpose(1, 0, 2)
    y = cali_y_train

    if args.eval_dataset_name is not None:
        eval_ccs_acc, eval_y_preds_pos, eval_y_preds_neg = fitted_ccs.get_acc(eval_neg_hs_test, eval_pos_hs_test, eval_y_test, return_conf=True)

        eval_cali_x, eval_cali_y = calibration_curve(eval_y_test, eval_y_preds_pos.flatten(), n_bins=10)
        if (eval_cali_x[0] < eval_cali_x[-1] and eval_cali_y[0] > eval_cali_y[-1]) or (eval_cali_x[0] > eval_cali_x[-1] and eval_cali_y[0] < eval_cali_y[-1]):
            #print('Flipping uncalibrated graph')
            eval_cali_y = eval_cali_y[::-1]
            temp = eval_y_preds_pos
            eval_y_preds_pos = eval_y_preds_neg
            eval_y_preds_neg = temp

        eval_ccs_brier = brier_score_loss(eval_y_test, eval_y_preds_pos.flatten())
        print(f"Eval CCS uncalcalibrated accuracy: {eval_ccs_acc}")
        print(f"Eval Brier score: {eval_ccs_brier}")
        all_results["eval_ccs_acc"] = eval_ccs_acc
        all_results["eval_ccs_brier"] = eval_ccs_brier


    # Calibration using unsupervised
    if args.use_dropout or args.use_dropout_loss:
        if args.use_dropout and args.use_dropout_loss:
            print("Calibrating with dropout and dropout loss")
        elif args.use_dropout:
            print("Calibrating with dropout")
        else:
            print("Calibrating with dropout loss")

        ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size,
                        verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay,
                        var_normalize=args.var_normalize, use_dropout=args.use_dropout, use_dropout_loss=args.use_dropout_loss,
                        dropout_loss_weight=args.dropout_loss_weight, dropout_factor=args.dropout_factor)

        calibrated_ccs = ccs.fit(None, None)
        ccs_acc, y_preds_pos, y_preds_neg = calibrated_ccs.get_acc(neg_hs_test, pos_hs_test, y_test, return_conf=True)
        cali_x, cali_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
        np.histogram(y_preds_pos.flatten(), bins=10, range=(0, 1))
        if (cali_x[0] < cali_x[-1] and cali_y[0] > cali_y[-1]) or (cali_x[0] > cali_x[-1] and cali_y[0] < cali_y[-1]):
            #print('Flipping uncalibrated graph')
            cali_y = cali_y[::-1]
            temp = y_preds_pos
            y_preds_pos = y_preds_neg
            y_preds_neg = temp

    #Calibration using BCE
    elif args.calibration_type == 'bce':
        print(f"Calibrating with {args.calibration_type}")
        calibrated_ccs = fitted_ccs.fit(X, y)
        ccs_acc, y_preds_pos, y_preds_neg = calibrated_ccs.get_acc(neg_hs_test, pos_hs_test, y_test, return_conf=True)
        cali_x, cali_y = calibration_curve(y_test, y_preds_pos.flatten(), n_bins=10)
        if (cali_x[0] < cali_x[-1] and cali_y[0] > cali_y[-1]) or (cali_x[0] > cali_x[-1] and cali_y[0] < cali_y[-1]):
            #print('Flipping calibrated graph')
            cali_y = cali_y[::-1]
            temp = y_preds_pos
            y_preds_pos = y_preds_neg
            y_preds_neg = temp
            #cali_x, cali_y = calibration_curve(y_test, y_preds_neg.flatten(), n_bins=10)

    # Calibration with sklearn
    elif args.calibration_type in ['sigmoid', 'isotonic']:
        print(f"Calibrating with {args.calibration_type}")
        calibrated_ccs = cccv(fitted_ccs, cv='prefit', method=args.calibration_type)
        calibrated_ccs.fit(X, y)
        X = np.stack([pos_hs_test, neg_hs_test], axis=0).transpose(1, 0, 2)
        y_preds = calibrated_ccs.predict_proba(X)
        y_preds_pos = y_preds[:,1]
        y_preds_neg = y_preds[:,0]
        predictions = (y_preds[:,1] < 0.5).astype(int)
        acc = (predictions == y_test).mean()
        ccs_acc = max(acc, 1 - acc)
        cali_x, cali_y = calibration_curve(y_test, y_preds[:,1].flatten(), n_bins=10)
        if (cali_x[0] < cali_x[-1] and cali_y[0] > cali_y[-1]) or (cali_x[0] > cali_x[-1] and cali_y[0] < cali_y[-1]):
            print('Flipping calibrated graph')
            cali_y = cali_y[::-1]
            temp = y_preds_pos
            y_preds_pos = y_preds_neg
            y_preds_neg = temp
            #cali_x, cali_y = calibration_curve(y_test, y_preds[:,0].flatten(), n_bins=10)
    cali_brier = brier_score_loss(y_test, y_preds_pos.flatten())
    all_results[f"{args.calibration_type}_acc"] = ccs_acc
    all_results[f"{args.calibration_type}_brier"] = cali_brier
    print(f"CCS calibrated accuracy: {ccs_acc}")
    print(f"Brier score: {cali_brier}")


    if args.eval_dataset_name is not None:
        X = np.stack([eval_pos_hs_test, eval_neg_hs_test], axis=0).transpose(1, 0, 2)
        eval_y_preds = calibrated_ccs.predict_proba(X)
        eval_y_preds_pos = eval_y_preds[:,1]
        eval_y_preds_neg = eval_y_preds[:,0]

        eval_predictions = (eval_y_preds[:,1] < 0.5).astype(int)
        eval_acc = (eval_predictions == eval_y_test).mean()
        eval_ccs_acc = max(eval_acc, 1 - eval_acc)

        #eval_ccs_acc, eval_y_preds_pos, eval_y_preds_neg = calibrated_ccs.get_acc(eval_neg_hs_test, eval_pos_hs_test, eval_y_test, return_conf=True)

        eval_cali_x, eval_cali_y = calibration_curve(eval_y_test, eval_y_preds_pos.flatten(), n_bins=10)
        if (eval_cali_x[0] < eval_cali_x[-1] and eval_cali_y[0] > eval_cali_y[-1]) or (eval_cali_x[0] > eval_cali_x[-1] and eval_cali_y[0] < eval_cali_y[-1]):
            #print('Flipping uncalibrated graph')
            eval_cali_y = eval_cali_y[::-1]
            temp = eval_y_preds_pos
            eval_y_preds_pos = eval_y_preds_neg
            eval_y_preds_neg = temp

        eval_ccs_brier = brier_score_loss(eval_y_test, eval_y_preds_pos.flatten())
        print(f"Eval CCS calibrated accuracy: {eval_ccs_acc}")
        print(f"Eval Brier score: {eval_ccs_brier}")
        all_results[f"eval_{args.calibration_type}_ccs_acc"] = eval_ccs_acc
        all_results[f"eval_{args.calibration_type}_ccs_brier"] = eval_ccs_brier

    
    plt.hist([uncali_y_preds_pos, y_preds_pos.flatten()], color=['b', 'g'], bins=10, range=[0, 1], edgecolor='b', label=['CCS', f'{args.calibration_type} CCS'])
    plt.legend(loc='upper left')
    plt.xlabel('Probability classes (intervals of 0.1)')
    plt.ylabel(f'Number of predictions per class (total: {generation_args.num_examples / 2})')
    plt.savefig(f"{name.replace('.png', '')}_histogram.png")
    plt.clf()


    # For each prompt, have one dictionary file
    path = f'./scores_imdb/{prompt_idx}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            curr_dict = pickle.load(f)
        all_results.update(curr_dict)
    with open(path, 'wb') as f:
        pickle.dump(all_results, f)


    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated: 0.0000')
    plt.plot(lr_y, lr_x, marker = '+', label = f'LR: {all_results["lr_brier"]:.4f}')
    plt.plot(uncali_y, uncali_x, marker = '.', label = f'CCS: {all_results["ccs_brier"]:.4f}')
    plt.plot(cali_y, cali_x, marker = 'x', label = f'{args.calibration_type} CCS: {cali_brier:.4f}')
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.savefig(name)



if __name__ == "__main__":
    parser = get_parser(evaluate=True)
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    cdn = generation_args.calibration_dataset_name
    del generation_args.calibration_dataset_name

    calibration_type = generation_args.calibration_type
    del generation_args.calibration_type
    use_dropout = generation_args.use_dropout
    del generation_args.use_dropout
    use_dropout_loss = generation_args.use_dropout_loss
    del generation_args.use_dropout_loss
    dropout_loss_weight = generation_args.dropout_loss_weight
    del generation_args.dropout_loss_weight
    dropout_factor = generation_args.dropout_factor
    del generation_args.dropout_factor
    confidence_loss_scale = generation_args.confidence_loss_scale
    del generation_args.confidence_loss_scale
    eval_dataset_name = generation_args.eval_dataset_name
    del generation_args.eval_dataset_name
    eval_prompt_idx = generation_args.eval_prompt_idx
    del generation_args.eval_prompt_idx

    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", type=bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")

    # Actively modulate these variables
    # These have been moved to utils.py
    #parser.add_argument("--calibration_type", type=str, default="sigmoid")
    #parser.add_argument("--use_dropout", type=bool, default=True)
    #parser.add_argument("--use_dropout_loss", type=bool, default=True)
    #parser.add_argument("--dropout_loss_weight", type=float, default=1.0)
    #parser.add_argument("--dropout_factor", type=float, default=0.1)

    args = parser.parse_args()
    args.calibration_dataset_name = cdn
    args.calibration_type = calibration_type
    args.use_dropout = use_dropout
    args.use_dropout_loss = use_dropout_loss
    args.dropout_loss_weight = dropout_loss_weight
    args.dropout_factor = dropout_factor
    args.confidence_loss_scale = confidence_loss_scale
    args.eval_dataset_name = eval_dataset_name
    args.eval_prompt_idx = eval_prompt_idx
    main(args, generation_args)
