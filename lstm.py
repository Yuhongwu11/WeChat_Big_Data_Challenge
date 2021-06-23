import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from comm import ACTION_LIST, STAGE_END_DAY
from evaluation import uAUC, compute_weighted_score

torch.manual_seed(2021)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# reading dataset
ROOT_PATH = "./data/"


def get_file(stage, action):

    if stage in ["online_train", "offline_train"]:
        action = action
    else:
        action = "all"

    file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=stage, action=action, day=STAGE_END_DAY[stage])
    stage_dir = os.path.join(ROOT_PATH, stage, file_name)
    df = pd.read_csv(stage_dir)

    return df


def load_data(stage, action, batch_size, shuffle):
    df = get_file(stage, action)

    embedding = nn.Embedding(len(df), 10)
    user_embedding = embedding(torch.from_numpy(df['userid'].values))
    feed_embedding = embedding(torch.from_numpy(df['feedid'].values))
    author_embedding = embedding(torch.from_numpy(df['authorid'].values))
    bgm_singer_embedding = embedding(torch.from_numpy(df['bgm_singer_id'].values))
    bgm_song_embedding = embedding(torch.from_numpy(df['bgm_song_id'].values))
    video_embedding = embedding(torch.from_numpy(df['videoplayseconds'].values).long())
    device_embedding = embedding(torch.from_numpy(df['device'].values).long())

    features = torch.stack(
            (user_embedding, feed_embedding, author_embedding, bgm_singer_embedding, bgm_song_embedding,
             video_embedding, device_embedding), dim=1)
    labels = torch.from_numpy(df[action].values)

    ds = TensorDataset(features, labels)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return ds_loader


class LSTM1(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout):
        super(LSTM1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.linear_1 = nn.Linear(hidden_dim*2, hidden_dim*1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        out, hidden = self.lstm(x)
        out = out.reshape(-1, self.hidden_dim*2)
        linear1_out = self.linear_1(out)
        linear1_out = self.relu(linear1_out)
        linear2_out = self.linear_2(linear1_out)
        sigmoid_out = self.sigmoid(linear2_out)
        sigmoid_out = sigmoid_out.reshape(batch_size, -1)
        sigmoid_out = sigmoid_out[:, -1]
        return sigmoid_out


def train(stage, action, model, criterion, optimizer, num_epochs, batch_size):
    ds_loader = load_data(stage, action, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        train_loss = []

        for i, dataset in enumerate(ds_loader):
            model.train()
            data = dataset[0].to(device)
            label = dataset[1].to(device)
            optimizer.zero_grad()
            output= model(data)
            loss = criterion(output, label.float())
            train_loss.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f'Batch {i}')


def evaluate(stage, action, model):
    model.eval()
    with torch.no_grad():
        df = get_file(stage, action)
        userid_list = df['userid'].astype(str).tolist()
        labels = df[action].values

        ds_loader = load_data(stage, action, batch_size=len(df), shuffle=False)
        for data, label in ds_loader:
            data = data.to(device)
            label = label.to(device)
            predicts = model(data)
            predicts_np = predicts.numpy()
            predicts_df = pd.DataFrame(predicts_np)

        uauc = uAUC(labels, predicts_df, userid_list)

    return df[["userid", "feedid"]], predicts_df, uauc


def predict(stage, action, model, t):
    model.eval()
    with torch.no_grad():
        df = get_file(stage, action)

        ds_loader = load_data(stage, action, batch_size=len(df), shuffle=False)
        for data, label in ds_loader:
            data = data.to(device)
            label = label.to(device)
            predicts = model(data)
            predicts_np = predicts.numpy()
            predicts_df = pd.DataFrame(predicts_np)

        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time() - t) * 1000.0 / len(df) * 2000.0

    return df[["userid", "feedid"]], predicts_df, ts


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def save_model(stage, action, model):
    PATH = "./data/model"
    model_checkpoint_stage_dir = os.path.join(PATH, stage, action)
    if not os.path.exists(model_checkpoint_stage_dir):
        # 如果模型目录不存在，则创建该目录
        os.makedirs(model_checkpoint_stage_dir)
    elif stage in ["online_train", "offline_train"]:
        # 训练时如果模型目录已存在，则清空目录
        del_file(model_checkpoint_stage_dir)
    torch.save(model, model_checkpoint_stage_dir)


def load_model(stage, action):
    if stage in ["evaluate", "offline_train"]:
        stage = "offline_train"
    else:
        stage = "online_train"
    PATH = "./data/model"
    model_checkpoint_stage_dir = os.path.join(PATH, stage, action)
    model = torch.load(model_checkpoint_stage_dir)

    return model


def main(argv):
    t = time.time()
    stage = argv[1]
    print('Stage: %s' % stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None

    # Hyper parameters
    num_epochs = 1
    batch_size = 128
    learning_rate = 0.01
    embedding_dim = 10
    hidden_dim = 128
    num_layers = 2
    output_dim = 1
    bidirectional = True
    dropout = 0.5

    model = LSTM1(embedding_dim, hidden_dim, num_layers, output_dim, bidirectional, dropout)
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for action in ACTION_LIST:
        print("Action:", action)

        if stage in ["online_train", "offline_train"]:
            train(stage, action, model, criterion, optimizer, num_epochs, batch_size)
            save_model(stage, action, model)
            ids, predicts, action_uauc = evaluate(stage, action, model)
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            model = load_model(stage, action)
            ids, predicts, action_uauc = evaluate(stage, action, model)
            eval_dict[action] = action_uauc
            predict_dict[action] = predicts

        if stage == "submit":
            model = load_model(stage, action)
            ids, predicts, ts = predict(stage, action, model, t)
            predict_dict[action] = predicts
            predict_time_cost[action] = ts

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(ROOT_PATH, stage, file_name)
        print('Save to: %s' % submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s' % (time.time()-t))


if __name__ == "__main__":
    main(sys.argv)