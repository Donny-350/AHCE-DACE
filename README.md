# A Robust Attention-based Cardinality Estimator with Domain Adaptation

A PyTorch Implementation of AHCE and DACE. We also provide the source code for data import, getting PostgreSQL cardinality estimation, and getting histogram information.

## Requirements

- Python 3.9.7
- PyTorch 1.10.2

## Dataset

The project works on PostgreSQL 11. If you do not have PostgreSQL installed, you first need to install [PostgreSQL](https://www.postgresql.org/download/). Our experiments are conducted on the [IMDB](https://www.imdb.com/interfaces/) and [Forest](https://archive.ics.uci.edu/ml/datasets/Covertype) datasets. You can use our forest.py script to import forest data into PostgreSQL, and similarly, we also provide a script to import Power data. 

Example usage:

```shell
cd importForestAndPower
python forest.py
```

Because the encoding part of our Attention-based Cardinality Estimator integrates histogram information. You can use a script we wrote to collect histogram information.

For example:

If you want to get the histogram information of Forest, you can run getForestHistogram.ipynb to get a histogram_forest.csv.

## Running experiments

### AHCE

If you want to train or test the performance of the AHCE model, you can verify its single-table and multi-table cardinality estimation performance on the two real datasets of forest and imdb.

**Parameter settings.**By default, batch_size is 256, the number of hidden layer neurons is 256, and epoch is 100.

You can test different workloads by adjusting the workload_name in the train_and_predict method in train.py. You can train and test AHCE performance using the following command.

```python
cd AHCE
python train.py
```

### DACE

When workload drift occurs, we can reduce q-error through the DACE framework. Among them, we can use two methods: distance-based MMD and dversarial-based GRL. For example, you can run it in the following format:

Distance-based MMD：

```python
cd DACE/DA(MMD)
# NoDA
python train.py
# Domain Adaptation through MMD method
python trainDA.py
```

Dversarial-based GRL：

```python
cd DACE/DA(GRL)
# NoDA
python train.py
# Domain Adaptation through GRL method
python trainDA.py
```

### LW-NN & PostgreSQL

In order to compare with LW-NN, we reproduced its code. You can run the train.py training model under the LW-NN folder, and test the model effect with test.py.

```shell
cd LW-NN
python train.py
python test.py
```

If you want to compare with traditional PostgreSQL, you can run get_cardinality_estimate_actual.py in the get-postgresql-cardinality folder, and then you will get a csv result file, in which you can see the estimated cardinality, actual cardinality, q-error, etc. information.

```shell
cd get-postgresql-cardinality
python get_cardinality_estimate_actual.py
```

## Code References:

- MSCN: [https://github.com/andreaskipf/learnedcardinalities](https://github.com/andreaskipf/learnedcardinalities)
- DANN: https://github.com/zengjichuan/DANN
- DeepDB: https://github.com/DataManagementLab/deepdb-public
- NeuroCard: https://github.com/neurocard/neurocard
- CEDA(Our Demo): https://github.com/Donny-350/CEDA

