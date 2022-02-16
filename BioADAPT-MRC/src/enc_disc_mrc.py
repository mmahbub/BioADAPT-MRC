import torch
import torch.nn as nn
from torch.autograd import Function
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import src.configs as configs

class Encoder(nn.Module):
    def __init__(self,
                 model,
                 freeze_encoder=False):
        super(Encoder, self).__init__()

        if configs.model_type == 'bert':
            self.encoder = model.bert
        elif configs.model_type == 'bart':
            self.encoder = model.model.encoder
        elif configs.model_type == 'electra':
            self.encoder = model.electra
        elif configs.model_type == 'distilbert':
            self.encoder = model.distilbert

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self,
                inputs):
        sequence_output = self.encoder(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

        return sequence_output


class QA_Out(nn.Module):
    def __init__(self,
                 model,
                 freeze_qa_output_generator=False
                 ):
        super(QA_Out, self).__init__()
        self.qa_outputs = model.qa_outputs

        if freeze_qa_output_generator:
            for param in self.qa_outputs.parameters():
                param.requires_grad = False

    def forward(self,
                encoder_out,
                start_positions=None,
                end_positions=None):

        logits = self.qa_outputs(encoder_out.last_hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # if we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=encoder_out.hidden_states,
            attentions=encoder_out.attentions,
        )


class Disc_QA_Out(nn.Module):
    def __init__(self,
                 freeze_aux_qa_output_generator=False, 
                 freeze_discriminator_encoder=False
                 ):
        super(Disc_QA_Out, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model = configs.emb_dim,
                                                   dim_feedforward = 3072,
                                                   dropout = 0.1,
                                                   activation = "gelu",
                                                   nhead = 12)
        self.discriminator_encoder = nn.TransformerEncoder(encoder_layer,
                                                           num_layers=1)

        if configs.USE_AUX_QA_LOSS:
            self.qa_outputs = nn.Linear(configs.emb_dim, 2)

        if freeze_discriminator_encoder:
            for param in self.discriminator_encoder.parameters():
                param.requires_grad = False

        if freeze_aux_qa_output_generator:
            if configs.USE_AUX_QA_LOSS:
                for param in self.qa_outputs.parameters():
                    param.requires_grad = False

    def forward(self,
                encoder_out,
                start_positions=None,
                end_positions=None):

        output = self.discriminator_encoder(encoder_out)
        
        # hidden representation of the CLS token, which is at position 0
        last_hidden_state_cls = output[:, 0, :]
        encoder_last_hidden_state_cls = encoder_out[:, 0, :]

        if not configs.USE_AUX_QA_LOSS:
            return last_hidden_state_cls, 0

        logits = self.qa_outputs(output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return last_hidden_state_cls, QuestionAnsweringModelOutput(loss=total_loss,
                                                                   start_logits=start_logits,
                                                                   end_logits=end_logits)

class ReverseLayerF(Function): 
     """
     Gradient Reversal Layer from:
     Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
     Forward pass is the identity function. In the backward pass,
     the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
     """
    @staticmethod
    def forward(ctx, x, _lambda_):
        ctx._lambda_ = _lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx._lambda_
        return output, None
