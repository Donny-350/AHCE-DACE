import argparse
import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ahce.util import *
from ahce.data import get_train_datasets, load_data, make_dataset
from ahce.model import AHCE
import numpy as np

SEED=20
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True

# 获取文件大小的函数
def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)  # 转换为MB
    return size_mb

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

        samples, predicates_hist, predicates, joins, targets, sample_masks, predicates_hist_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates_hist, predicates, joins, targets = samples.cuda(), predicates_hist.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicates_hist_masks, predicate_masks, join_masks = sample_masks.cuda(), predicates_hist_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates_hist, predicates, joins, targets = Variable(samples), Variable(predicates_hist), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicates_hist_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicates_hist_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, predicates_hist, predicates, joins, sample_masks, predicates_hist_masks, predicate_masks, join_masks)
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
    return qerror


def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    # Load training and validation data
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_hist_feats = 50
    predicate_feats = len(column2vec) + len(op2vec) + 1  #13
    print("predicate_feats: ", predicate_feats)
    join_feats = len(join2vec)

    model = AHCE(sample_feats, predicate_hist_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    val_data_loader = DataLoader(test_data, batch_size=batch_size)

    # Load test data
    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    hist_file = get_hist_file("data/histogram_string.csv")
    hist_list = [hist_file['table_column'][i] for i in range(len(hist_file))]
    hist_map = {}
    for i in range(len(hist_file)):
        table_column = hist_file['table_column'][i]
        bins = hist_file['bins'][i]
        hist_map[table_column] = bins

    predicates_hist = []
    for i in range(len(predicates)):
        predicate_list = []
        for predicate in predicates[i]:
            list1 = getPredicateHistEncode(predicate, hist_list, hist_map)
            predicate_list.append(list1)
        predicates_hist.append(predicate_list)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_hist, predicates_test, joins_test, labels_test, max_num_joins,
                             max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    #存储loss
    loss_list = []
    timeTrainStart = time.time()

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):

            samples, predicates_hist, predicates, joins, targets, sample_masks, predicate_hist_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates_hist, predicates, joins, targets = samples.cuda(), predicates_hist.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_hist_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_hist_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates_hist, predicates, joins, targets = Variable(samples), Variable(predicates_hist), Variable(predicates), Variable(joins), Variable(
                targets)
            sample_masks, predicate_hist_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_hist_masks), Variable(predicate_masks), Variable(
                join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates_hist, predicates, joins, sample_masks, predicate_hist_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
        loss_list.append(loss_total / len(train_data_loader))

        preds_test, t_total = predict(model, test_data_loader, cuda)
        print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

        # Unnormalize
        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

        # Print metrics
        print("\nQ-Error " + workload_name + ":")
        qerror = print_qerror(preds_test_unnorm, label)
        print("qerror的长度: ", len(qerror))

    timeTrainEnd = time.time()
    print("Ours模型训练时间为：", timeTrainEnd-timeTrainStart)

    # 保存模型
    model_save_path = "model/AHCE_model_256.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # 打印模型大小
    model_size = get_file_size(model_save_path)
    print("Model size: {:.2f} MB".format(model_size))

    # loss_list = np.array(loss_list)
    # print("loss的长度：", len(loss_list))
    # np.savez("loss/AHCE_job_light.npz", array_name=loss_list)

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_val, t_total = predict(model, val_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_val, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("testset", help="synthetic, scale, or job-light")
#     parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
#     parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
#     parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
#     parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
#     parser.add_argument("--cuda", help="use CUDA", action="store_true")
#     args = parser.parse_args()
#     train_and_predict(args.testset, args.queries, args.epochs, args.batch, args.hid, args.cuda)


if __name__ == "__main__":
    train_and_predict("synthetic", 100000, 100, 256, 256, True)
