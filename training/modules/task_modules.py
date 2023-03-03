import torch


class GlueClassification(torch.nn.Module):
    def __init__(self, model_dim, num_classes):
        super(GlueClassification, self).__init__()
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.pooler_layer = torch.nn.Linear(model_dim, model_dim)
        self.fc_layer = torch.nn.Linear(model_dim, num_classes)

    def forward(self, hidden_states, pooler_index=0):
        pooled = hidden_states[:, pooler_index, :]
        pooled = self.pooler_layer(pooled)
        pooled = torch.tanh(pooled)
        return self.fc_layer(pooled)
