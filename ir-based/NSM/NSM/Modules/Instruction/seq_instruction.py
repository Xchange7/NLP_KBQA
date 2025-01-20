import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Instruction.base_instruction import BaseInstruction
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class LSTMInstruction(BaseInstruction):
    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)

        self.word_embedding = word_embedding
        self.num_word = num_word

        # self.projection = nn.Linear(768, 2 * self.entity_dim).to(self.device) # 40 X 100
        self.projection = nn.Linear(768, 50).to(self.device)  # 40 X 100
        self.projection_po = nn.Linear(768, 50).to(self.device)
        self.projection_qhe = nn.Linear(768, 50).to(self.device)

        # 加载 BERT 预训练模型和 tokenizer
        self.bert_model_name = "bert-base-uncased"  # 可以替换成自己的模型路径
        self.bert = BertModel.from_pretrained(self.bert_model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)

        self.entity_dim = self.entity_dim  # 从 BaseInstruction 继承的参数
        self.cq_linear = nn.Linear(in_features=2 * self.entity_dim, out_features=self.entity_dim).to(self.device)
        # self.cq_linear = nn.Linear(in_features = 768,  out_features=50, bias=True )
        self.ca_linear = nn.Linear(in_features=self.entity_dim, out_features=1).to(self.device)

        # 添加多个线性层，用于不同步骤的处理，保持与 LSTMInstruction 一致
        for i in range(self.num_step):
            self.add_module(f'question_linear{i}', nn.Linear(self.entity_dim, self.entity_dim).to(self.device))

    def encode_question(self, query_text):
        # 检查是否是Tensor，如果是则转换为Python列表
        if isinstance(query_text, torch.Tensor):
            query_text = query_text.cpu().numpy().tolist()

        # 检查是否是列表类型的数字（原始输入是索引）
        if isinstance(query_text, list) and isinstance(query_text[0], list):
            # 假设 query_text 是索引列表，需要将其转换为字符串
            query_text = [" ".join(map(str, text)) for text in query_text]
        elif isinstance(query_text, list) and isinstance(query_text[0], int):
            query_text = [" ".join(map(str, query_text))]  # 处理单条数据

        inputs = self.tokenizer(query_text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        query_hidden_emb = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        pooled_output = outputs.pooler_output  # (batch_size, 768)

        pooled_output = self.projection_po(pooled_output)
        query_hidden_emb = self.projection_qhe(query_hidden_emb)

        # self.query_node_emb = self.cq_linear(pooled_output).unsqueeze(1)  # 映射到 (batch_size, 1, 50)
        self.query_node_emb = pooled_output.unsqueeze(1)
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = attention_mask.float()

        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        """
        初始化推理过程，编码问题并设置初始状态。
        """
        batch_size = query_text.size(0)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb[:, 0, :]  # 取第一个 token 代表整体

        # self.projection = nn.Linear(768, 2 * self.entity_dim).to(self.device) # 40 X 100
        # query_hidden_emb = self.projection(query_hidden_emb)

        # query_hidden_emb = self.cq_linear(query_hidden_emb)  # 映射到 100 维

        if query_node_emb is None:
            query_node_emb = self.query_node_emb

        relational_ins = relational_ins.unsqueeze(1)  # (batch_size, 1, hidden_size)
        question_linear = getattr(self, f'question_linear{step}')
        q_i = question_linear(self.linear_drop(query_node_emb))

        # print("Relational_ins: ", relational_ins.shape)
        # q_i = self.projection(q_i)

        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb.unsqueeze(1)))

        attn_weight = F.softmax(ca + (1 - self.query_mask.unsqueeze(2)) * VERY_SMALL_NUMBER, dim=1)
        relational_ins = torch.sum(attn_weight * self.query_hidden_emb, dim=1)

        return relational_ins, attn_weight

    def forward(self, query_text):
        """
        前向传播流程：使用 BERT 处理文本，结合推理步骤获取指令。

        :param query_text: (batch_size, seq_len) 的文本输入
        :return: 生成的推理指令，注意力权重
        """
        self.init_reason(query_text)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins

        return self.instructions, self.attn_list
    # def __init__(self, args, word_embedding, num_word):
    #     super(LSTMInstruction, self).__init__(args)
    #     self.word_embedding = word_embedding
    #     self.num_word = num_word
    #     self.encoder_def()
    #     entity_dim = self.entity_dim
    #     self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
    #     self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
    #     for i in range(self.num_step):
    #         self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
    #
    # def encoder_def(self):
    #     # initialize entity embedding
    #     word_dim = self.word_dim
    #     kg_dim = self.kg_dim
    #     kge_dim = self.kge_dim
    #     entity_dim = self.entity_dim
    #     self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
    #                                 batch_first=True, bidirectional=False)
    #
    # def encode_question(self, query_text):
    #     batch_size = query_text.size(0)
    #     query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
    #     query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
    #                                                      self.init_hidden(1, batch_size,
    #                                                                       self.entity_dim))  # 1, batch_size, entity_dim
    #     self.instruction_hidden = h_n
    #     self.instruction_mem = c_n
    #     self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
    #     self.query_hidden_emb = query_hidden_emb
    #     self.query_mask = (query_text != self.num_word).float()
    #     return query_hidden_emb, self.query_node_emb
    #
    # def init_reason(self, query_text):
    #     batch_size = query_text.size(0)
    #     self.encode_question(query_text)
    #     self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
    #     self.instructions = []
    #     self.attn_list = []
    #
    # def get_instruction(self, relational_ins, step=0, query_node_emb=None):
    #     query_hidden_emb = self.query_hidden_emb
    #     query_mask = self.query_mask
    #     if query_node_emb is None:
    #         query_node_emb = self.query_node_emb
    #     relational_ins = relational_ins.unsqueeze(1)
    #     question_linear = getattr(self, 'question_linear' + str(step))
    #     q_i = question_linear(self.linear_drop(query_node_emb))
    #     cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
    #     # batch_size, 1, entity_dim
    #     ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
    #     # batch_size, max_local_entity, 1
    #     # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
    #     attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
    #     # batch_size, max_local_entity, 1
    #     relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
    #     return relational_ins, attn_weight
    #
    # def forward(self, query_text):
    #     self.init_reason(query_text)
    #     for i in range(self.num_step):
    #         relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
    #         self.instructions.append(relational_ins)
    #         self.attn_list.append(attn_weight)
    #         self.relational_ins = relational_ins
    #     return self.instructions, self.attn_list
    #
    # # def __repr__(self):
    # #     return "LSTM + token-level attention"

