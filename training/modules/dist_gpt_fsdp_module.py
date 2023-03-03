import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from .task_modules import GlueClassification
from .gpt_modules import MultiHeadAttention, TwoLayerMLP, GPTEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper


# This is only implemented to support checkpoint in FSDP

class GPTTransformerFsdpLayer(torch.nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048, layer_norm_eps=1e-5, use_checkpoint=True,
                 explicit_fsdp=False) -> None:
        super(GPTTransformerFsdpLayer, self).__init__()
        self.attn = MultiHeadAttention(model_dim, head_num)
        if use_checkpoint:
            self.attn = checkpoint_wrapper(self.attn)
        if explicit_fsdp:
            self.attn = FSDP(self.attn, reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                             flatten_parameters=False)
        # Implementation of Feedforward model
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        if use_checkpoint:
            self.mlp = checkpoint_wrapper(self.mlp)
        if explicit_fsdp:
            self.attn = FSDP(self.attn, reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                             flatten_parameters=False)
        self.norm1 = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2))
        x.requires_grad_(True)
        x = self.attn(x)
        x = self.norm2(x)
        # x = x + self.dropout_2(self.ff(x2))
        x.requires_grad_(True)
        x = self.mlp(x)
        return x


class GPTGlueFsdpModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True):
        super(GPTGlueFsdpModel, self).__init__()
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerFsdpLayer(args.embedding_dim, args.num_heads,
                                                       args.embedding_dim * 4, use_checkpoint, explicit_fsdp=False))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = GlueClassification(args.embedding_dim, num_classes)

    def forward(self, input_ids, position_ids):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)


class GPTFsdpStageBase(torch.nn.Module):
    def __init__(self, args, num_stage_layers, vocab_size, num_classes, use_checkpoint=True, explicit_fsdp=True):
        super(GPTFsdpStageBase, self).__init__()
        self._vocab_size = vocab_size
        self._explicit_fsdp = explicit_fsdp
        self._use_checkpoint = use_checkpoint
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
        self._num_classes = num_classes
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = num_stage_layers

    def _create_first_layer(self):
        emb = GPTEmbedding(self._vocab_size, self._embedding_dim, self._seq_length)
        if self._explicit_fsdp:
            return FSDP(emb, reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                        flatten_parameters=False)
        else:
            return emb

    def _create_last_layer(self):
        classifier = GlueClassification(self._embedding_dim, self._num_classes)
        if self._explicit_fsdp:
            return FSDP(classifier, reshard_after_forward=True, move_params_to_cpu=False, mixed_precision=False,
                        flatten_parameters=False)
        else:
            return classifier

    def _create_fsdp_transformer_layer(self):
        return GPTTransformerFsdpLayer(self._embedding_dim, self._num_heads, self._feedforward_dim,
                                       use_checkpoint=self._use_checkpoint, explicit_fsdp=self._explicit_fsdp)


class GPTFsdpStageFirst(GPTFsdpStageBase):
    def __init__(self, args, num_stage_layers, vocab_size, num_classes, device, use_checkpoint=True, explicit_fsdp=True):
        super(GPTFsdpStageFirst, self).__init__(args, num_stage_layers, vocab_size, num_classes, use_checkpoint,
                                                explicit_fsdp)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_fsdp_transformer_layer())
        self.model = torch.nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x)
        return out


class GPTFsdpStageMiddle(GPTFsdpStageBase):
    def __init__(self, args, num_stage_layers, vocab_size, num_classes, device, use_checkpoint=True, explicit_fsdp=True):
        super(GPTFsdpStageMiddle, self).__init__(args, num_stage_layers, vocab_size, num_classes, use_checkpoint,
                                                 explicit_fsdp)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_fsdp_transformer_layer())
        self.model = torch.nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x)
        return out


class GPTFsdpStageLast(GPTFsdpStageBase):
    def __init__(self, args, num_stage_layers, vocab_size, num_classes, device, use_checkpoint=True, explicit_fsdp=True):
        super(GPTFsdpStageLast, self).__init__(args, num_stage_layers, vocab_size, num_classes, use_checkpoint,
                                               explicit_fsdp)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_fsdp_transformer_layer())
        module_list.append(self._create_last_layer())
        self.model = torch.nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x)
        return out
