import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.nn as dglnn
from dgl.nn.pytorch import edge_softmax, GATConv
from sklearn import metrics
import numpy as np
import pandas as pd
from tqdm import tqdm
np.random.seed(2022)#固定随机数种字

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h   #一次性为所有节点类型的 'h'赋值
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class Model(nn.Module):
    def __init__(self, in_feats_1, in_feats_2, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({'click': dglnn.GraphConv(in_feats_1, hid_feats),
                                            'click_by': dglnn.GraphConv(in_feats_2, hid_feats)}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({'click': dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='mean'),
                                            'click_by': dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='mean')})
        self.mlp = MLPPredictor(hid_feats, out_feats)

    def forward(self, graph, inputs, etype):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = self.mlp(graph, h, etype)
        h = torch.sigmoid(h)
        return h





#---------------开始加载数据处理训练--------------------------
graph_df = pd.read_csv('graph_data.csv')
user_id = torch.tensor(graph_df['user_id'].values).to(device)
item_id = torch.tensor(graph_df['item_id'].values).to(device)
graph_data = {
    ('user', 'click', 'item'): (user_id, item_id),
    ('item', 'click_by', 'user'): (item_id, user_id)
}

#获取节点的数目
n_user = max(user_id) + 1
n_item = max(item_id) + 1
#获取节点的特征维度
n_user_feat = len(graph_df['user_feat'].values[0].split(' '))
n_item_feat = len(graph_df['item_feat'].values[0].split(' '))
#生成特征
user_feat = np.zeros((n_user, n_user_feat))
item_feat = np.zeros((n_item, n_item_feat))
print('加载节点特征.....')
for _user_id, _item_id, _label, _user_feat, _item_feat in tqdm(graph_df.values):
    #对应节点特征赋值
    user_feat[_user_id] = np.array(_user_feat.split(' ')).astype(dtype=float)
    item_feat[_item_id] = np.array(_item_feat.split(' ')).astype(dtype=float)

label = graph_df['label'].values


print('加载训练集、验证集和测试集.....')
train_mask = []
val_mask = []
test_mask = []
for val in tqdm(label):
    if int(val) != -1:
        key = np.random.randint(0,100)
        if key <= 80:
            train_mask.append(True)
            val_mask.append(False)
        else:
            train_mask.append(False)
            val_mask.append(True)
        test_mask.append(False)
    else:
        train_mask.append(False)
        val_mask.append(False)
        test_mask.append(True)

g = dgl.heterograph(graph_data)
g.nodes['user'].data['feat'] = torch.tensor(user_feat, dtype=torch.float32).to(device)
g.nodes['item'].data['feat'] = torch.tensor(item_feat, dtype=torch.float32).to(device)
g.edges['click'].data['label'] = torch.tensor(label, dtype=torch.float32).to(device)
g.edges['click'].data['train_mask'] = torch.tensor(train_mask, dtype=torch.bool).to(device)
g.edges['click'].data['val_mask'] = torch.tensor(val_mask, dtype=torch.bool).to(device)
g.edges['click'].data['test_mask'] = torch.tensor(test_mask, dtype=torch.bool).to(device)



user_feats = g.nodes['user'].data['feat']
item_feats = g.nodes['item'].data['feat']
node_features = {'user': user_feats, 'item': item_feats}
train_mask = g.edges['click'].data['train_mask']
val_mask = g.edges['click'].data['val_mask']
test_mask = g.edges['click'].data['test_mask']
label = g.edges['click'].data['label']


model = Model(n_user_feat, n_item_feat, 100, 1, g.etypes)
model.to(device)
opt = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()#定义二分类交叉熵损失函数


best_val_auc = 0
best_test_auc = 0
epoch_num = 100
for epoch in range(epoch_num):
    model.train()
    pre = model(g, node_features, 'click')
    train_loss = criterion(pre[train_mask], label[train_mask].reshape(-1, 1))
    val_loss = criterion(pre[val_mask], label[val_mask].reshape(-1, 1))
    train_pre = pre[train_mask].detach().cpu().numpy()
    val_pre = pre[val_mask].detach().cpu().numpy()
    train_auc = metrics.roc_auc_score(label[train_mask].cpu().numpy(), train_pre.reshape(1,-1)[0])
    val_auc = metrics.roc_auc_score(label[val_mask].cpu().numpy(), val_pre.reshape(1,-1)[0])
    opt.zero_grad()
    train_loss.backward()
    opt.step()


    if val_auc > best_val_auc:#选择验证集最优时候的模型进行预测
        test_pre = pre[test_mask].detach().cpu().numpy().reshape(1,-1)[0]
    print('epoch=',epoch,' train_loss=',train_loss.item(), ' val_loss=',val_loss.item(),' train_auc=',train_auc,' val_auc=',val_auc)

