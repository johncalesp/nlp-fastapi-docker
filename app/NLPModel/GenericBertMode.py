import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, distilBert):
        super(BERT_Arch, self).__init__()
        self.distilBert = distilBert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (output layer)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        last_hidden_state = self.distilBert(sent_id, attention_mask=mask).last_hidden_state[:, 0, :]

        # print(type(out))
        x = self.fc1(last_hidden_state)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer

        x = self.fc2(x)

        return x