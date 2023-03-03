from torch import nn
from .deberta_modules import DebertaV2Embeddings, DebertaV2Layers, DebertaClassificationHead


class DebertaStageBase(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self._to_cpu = False # (args.dist_backend == "gloo")
        self.config = config

    def _create_first_layer(self):
        return DebertaV2Embeddings(self.config)

    def _create_last_layer(self):
        return DebertaClassificationHead(self.config)

    def _create_transformer_layers(self, first_block=False):
        return DebertaV2Layers(self.config, first_block=first_block) # TODO: checkpoint


class DebertaStageFirst(DebertaStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        self.embeddings = self._create_first_layer().to(device)
        self.encoder = self._create_transformer_layers(first_block=True).to(device)

    def forward(self, x, token_type_ids=None, attention_mask=None):
        if self._to_cpu:
            x = x.to(self.device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        x = self.embeddings(x, token_type_ids=token_type_ids)
        out = self.encoder(x, attention_mask=attention_mask)
        return out.cpu() if self._to_cpu else out


class DebertaStageMiddle(DebertaStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        self.encoder = self._create_transformer_layers(first_block=False).to(device)

    def forward(self, x, attention_mask=None):
        if self._to_cpu:
            x = x.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        out = self.encoder(x, attention_mask=attention_mask)
        return out.cpu() if self._to_cpu else out


class DebertaStageLast(DebertaStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        self.encoder = self._create_transformer_layers(first_block=False).to(device)
        self.output_head = self._create_last_layer().to(device)

    def forward(self, x, attention_mask=None, input_ids=None):
        if self._to_cpu:
            x = x.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        x = self.encoder(x, attention_mask=attention_mask)
        out = self.output_head(x)
        return out.cpu() if self._to_cpu else out