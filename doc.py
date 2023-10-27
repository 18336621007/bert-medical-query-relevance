from transformers import get_scheduler
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, BertTokenizerFast
from transformers import get_scheduler
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate
import json
train_path = "./data/KUAKE-QQR_train.json"
test_path = "./data/KUAKE-QQR_dev.json"
save_path = "./model/model.pth"
#加载数据
def load_train_file(file_path):
    with open(file_path,"r",encoding="utf-8") as input_data:
        ids = []
        querys1 = []
        querys2 = []
        labels = []
        json_content = json.load(input_data)
        #逐条读取记录
        for block in json_content:
            query1 = block['query1']
            query2 = block['query2']
            label = block['label']

            querys1.append(query1)
            querys2.append(query2)
            labels.append(int(label.strip()))
        return querys1,querys2,labels
#加载验证数据
def load_test_file(file_path):
    with open(file_path,"r",encoding="utf-8") as input_data:
        querys1 = []
        querys2 = []
        labels = []
        json_content = json.load(input_data)
        #逐条读取记录
        for block in json_content:
            query1 = block['query1']
            query2 = block['query2']
            label = block['label']

            querys1.append(query1)
            querys2.append(query2)
            if label:
                labels.append(int(label.strip()))
            else:
                labels.append(0)

        return querys1,querys2,labels

train_querys1,train_querys2,train_labels = load_train_file(train_path)
print("获取到",len(train_querys1),"条训练数据")
test_querys1,test_querys2,test_labels = load_test_file(test_path)
print("获取到",len(test_querys1),"条测试数据")

#数据预处理，转化成Bert模型接受的格式
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
#'input_ids': [101, 1921, 817, 6783, 3890, 6589, 102, 6783, 3890, 817, 3419, 102],
#              [CLS] 天 价 输 液 费 [SEP] 输 液 价 格 [SEP]

class MyDataset(Dataset):
    def __init__(self,querys1,querys2,labels,tokenizer):
        self.querys1 = querys1
        self.querys2 = querys2
        if labels:
            self.labels = labels
        else:
            self.labels = 10
        self.tokenizer = tokenizer

        self.encode_querys = tokenizer(querys1,querys2,padding=True)

    def __len__(self):
        return len(self.querys1)

    def __getitem__(self, idx):#
        input_ids = torch.LongTensor(self.encode_querys['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encode_querys['attention_mask'][idx])
        labels = torch.LongTensor([self.labels[idx]])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataset = MyDataset(train_querys1,train_querys2,train_labels,tokenizer)
test_dataset = MyDataset(test_querys1,test_querys2,test_labels,tokenizer)
# print(train_dataset.__getitem__(1))
# print(test_dataset.__getitem__(1))

#定义模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
#model = torch.load(save_path)
# 定义Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64,shuffle=True)
print("训练数据封装成批次大小为",train_dataloader.batch_size,"的批次，共",len(train_dataloader),"步")

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
print("设定训练",num_epochs,"个周期")
num_training_steps = num_epochs * len(train_dataloader)
print("总步数:",num_training_steps)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#如果您可以访问GPU，请指定要使用GPU的设备。否则，在CPU上进行训练可能需要几个小时，而不是几分钟
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#训练model
progress_bar = tqdm(range(num_training_steps))
# model.train()
# for epoch in range(num_epochs):
#     for step,batch in enumerate(train_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}#字典{inputs_id:[],attention_mask:[],labels:[]}
#         outputs = model(**batch)#[batch_size,sequence_len,num_labels]
#         loss = outputs.loss
#         loss.backward()
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)#更新已经处理一个批次了
#         if step % 100 == 0:
#             print(f'Step {step} / {num_training_steps} - Training loss: {loss}')
# torch.save(model, save_path)

#
# #
# # # 评估函数
# #

# #
model.eval()
total = len(test_querys1)
acc = 0
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    for label,prediction in zip(batch['labels'],predictions):
        if(label == prediction):
            acc = acc + 1

print(f"准确率:{(acc / total):.2f}")


#写入数据
# goal_path = "./data/KUAKE-QQR_test.json"
# goal_querys1,goal_querys2,goal_labels = load_test_file(goal_path)
# goal_dataset = MyDataset(goal_querys1,goal_querys2,goal_labels,tokenizer)
# goal_dataloader = DataLoader(goal_dataset, batch_size=64)
#
#
# i = 1
# data = []
# for batch in goal_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch);
#
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#
#     for query, prediction in zip(batch['input_ids'], predictions):
#         firstOk = 0
#         #找出两个[SEP]的索引值
#         for index, value in enumerate(query):
#             if value == 102:
#                 if firstOk == 0:
#                     index1 = index
#                     firstOk = 1
#                 else:
#                     index2 = index
#         id = f"s{i}"
#         query1 = tokenizer.decode(query[1:index1]).replace(" ", "")
#         query2 = tokenizer.decode(query[index1+1:index2]).replace(" ", "")
#         label = prediction.tolist()
#
#         i += 1
#         #构建json数据
#         jdata = {
#             "id":id,
#             "query1":query1,
#             "query2":query2,
#             "label":label
#         }
#         data.append(jdata)
# # 使用 json.dumps 将 Python 字典序列化为 JSON 格式的字符串
# json_data = json.dumps(data, indent=2, ensure_ascii=False)  # indent 参数可选，用于美化 JSON 输出
# with open('./data/KUAKE-QQR_test_pred.json', 'w', encoding="utf-8") as output_data:
#     output_data.write(json_data)
#
#
# print(f"所有JSON 数据已写入到 {goal_path}")


