import argparse
import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from GRL.util import *
from GRL.data import get_train_datasets, load_data, make_dataset
from GRL.model import AHCE
import numpy as np

SEED=0
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        hid, outputs, _ = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("25th percentile: {}".format(np.percentile(qerror, 25)))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    # Load training and validation data
    timeStart = time.time()
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, max_num_joins, max_num_predicates, train_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)
    # sample_feats = 10
    # predicate_feats = 10 + 5 + 1
    # join_feats = 10
    model = AHCE(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    file_name = "workloads/target/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)

    # train and test data
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    print("the length of train_data_loader: ", len(train_data_loader))
    print("the length of test_data_loader: ", len(test_data_loader))

    max_batches = min(len(train_data_loader), len(test_data_loader))
    loss_fn_domain = torch.nn.NLLLoss()

    for epoch_idx in range(num_epochs):
        loss = 0.
        dl_source_iter = iter(train_data_loader)
        dl_target_iter = iter(test_data_loader)

        for batch_idx in range(max_batches):
            optimizer.zero_grad()
            p = float(batch_idx + epoch_idx * max_batches) / (num_epochs * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # Train on source domain
            samples_train, predicates_train, joins_train, targets_train, sample_masks_train, predicate_masks_train, join_masks_train = next(dl_source_iter)
            #samples_train, predicates_train, joins_train, targets_train, sample_masks_train, predicate_masks_train, join_masks_train = samples_train.cuda(), predicates_train.cuda(), joins_train.cuda(), targets_train.cuda(), sample_masks_train.cuda(), predicate_masks_train.cuda(), join_masks_train.cuda()
            hid_train, outputs, domain_pred = model(samples_train, predicates_train, joins_train, sample_masks_train,
                                         predicate_masks_train, join_masks_train, grl_lambda)
            y_s_domain = torch.zeros(batch_size, dtype=torch.long)
            #y_s_domain = y_s_domain.cuda()
            loss_regression = qerror_loss(outputs, targets_train.float(), min_val, max_val)
            loss_s_domain = loss_fn_domain(domain_pred, y_s_domain)

            # Train on target domain
            samples_test, predicates_test, joins_test, targets_test, sample_masks_test, predicate_masks_test, join_masks_test = next(dl_target_iter)
            #samples_test, predicates_test, joins_test, targets_test, sample_masks_test, predicate_masks_test, join_masks_test = samples_test.cuda(), predicates_test.cuda(), joins_test.cuda(), targets_test.cuda(), sample_masks_test.cuda(), predicate_masks_test.cuda(), join_masks_test.cuda()
            y_t_domain = torch.ones(batch_size, dtype=torch.long)
            #y_t_domain = y_t_domain.cuda()
            hid_test, _, domain_pred_test = model(samples_test, predicates_test, joins_test, sample_masks_test,
                                        predicate_masks_test,
                                        join_masks_test)
            loss_t_domain = loss_fn_domain(domain_pred_test, y_t_domain)
            loss = loss_regression + loss_s_domain + loss_t_domain
            loss.backward()
            optimizer.step()

            print("Epoch {}, loss: {}".format(epoch_idx, loss / len(train_data_loader)))

    timeEnd = time.time()
    print("训练时间DA：", timeEnd - timeStart)

    preds_train, t_total = predict(model, train_data_loader, False)
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    preds_test, t_total = predict(model, test_data_loader, False)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))
    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    # Print metrics
    print_qerror(preds_test_unnorm, label)



def main():
    train_and_predict('job-light', 100, 100, 70, 256, False)

if __name__ == "__main__":
    main()
