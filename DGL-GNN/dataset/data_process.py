import numpy as np
from tqdm import tqdm
np.random.seed(2022)#固定随机数种字

n_user = 10000#用户的ID范围
n_item = 100000#商品的ID范围
n_edge = 50000#边的数目
n_user_feat = 100#用户的特征维度
n_item_feat = 200#商品的特征维度

user_id = np.random.choice(n_user, n_edge)#在0-n_user之间随机生成n_edge个数字
item_id = np.random.choice(n_item, n_edge, replace=False)#在0-n_item之间随机生成n_edge个不重复的数字

user_feat = np.random.random((n_user, n_user_feat))
item_feat = np.random.random((n_item, n_item_feat))

label = []#-1表示不确定，0表示不购买，1表示购买
for i in range(n_edge):
    key = np.random.randint(0, 100)
    if key <= 10:
        label.append(-1)
    elif key <= 50:
        label.append(0)
    else:
        label.append(1) 


fp = open('graph_data.csv','w',encoding='utf-8')
fp.writelines('user_id,item_id,label,user_feat,item_feat\n')
for _user_id, _item_id, _label, _user_feat, _item_feat in tqdm(zip(user_id, item_id, label, user_feat, item_feat)):
    single_user_feat = ''
    single_item_feat = ''
    for feat in _user_feat:
        single_user_feat += str(feat) + ' '#空格作为分隔符
    for feat in _item_feat:
        single_item_feat += str(feat) + ' '#空格作为分隔符
    fp.writelines(str(_user_id) + ',' + str(_item_id) + ',' + str(_label) + ',' + single_user_feat[0:-1] + ',' + single_item_feat[0:-1] + '\n')
fp.close()




