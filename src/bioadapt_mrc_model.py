import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering
)
from src.enc_disc_mrc import (
    Encoder,
    Disc_QA_Out,
    QA_Out,
    ReverseLayerF
)
import src.configs as configs


class bioadapt_mrc_net(nn.Module):
    def __init__(self):
        super(bioadapt_mrc_net, self).__init__()

        configs.model_type = configs.model_type.lower()
        config = AutoConfig.from_pretrained(
            configs.config_name if configs.config_name else configs.pretrained_model_name_or_path,
            cache_dir=configs.cache_dir if configs.cache_dir else None,
        )

        pretrained_model = AutoModelForQuestionAnswering.from_pretrained(
            configs.pretrained_model_name_or_path,
            from_tf=bool(".ckpt" in configs.pretrained_model_name_or_path),
            config=config,
            cache_dir=configs.cache_dir if configs.cache_dir else None,
        )

        self.encoder = Encoder(pretrained_model,
                               freeze_encoder=configs.freeze_encoder)

        self.factoid_qa_output_generator = QA_Out(pretrained_model,
                                                  freeze_qa_output_generator=configs.freeze_qa_output_generator)

        self.disc_aux_qa_layer = Disc_QA_Out(freeze_aux_qa_output_generator=configs.freeze_aux_qa_output_generator, 
                                             freeze_discriminator_encoder=configs.freeze_discriminator_encoder)

        self.disc_aux_qa_layer.apply(self.init_weights)

    def forward(self, data_batch):
        question_context_features, start_positions, end_positions = self.move_to_cuda(data_batch)

        encodings = self.siamese_encoder(question_context_features)
        factoid_qa_outputs, original_qa_loss = self.siamese_factoid_qa_out_gen(encodings,
                                                                               start_positions,
                                                                               end_positions)
        adv_loss = None
        aux_qa_loss = None
        aux_qa_outputs = None
        total_loss = configs.qa_loss_alpha * original_qa_loss
        if configs.do_train:
            adv_loss, aux_qa_outputs, aux_qa_loss = self.siamese_discriminator(encodings,
                                                                               start_positions,
                                                                               end_positions)
            total_loss += configs.adv_loss_beta * adv_loss + \
                          configs.aux_layer_gamma * aux_qa_loss
            
        return encodings, factoid_qa_outputs, aux_qa_outputs, adv_loss, aux_qa_loss, original_qa_loss, total_loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=configs.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=configs.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def move_to_cuda(self, data_batch):
        question_context_features, start_positions, end_positions = data_batch
        for key in question_context_features[0]:
            question_context_features[0][key] = torch.autograd.Variable(
                question_context_features[0][key].to(configs.device).long())
            question_context_features[1][key] = torch.autograd.Variable(
                question_context_features[1][key].to(configs.device).long())
            question_context_features[2][key] = torch.autograd.Variable(
                question_context_features[2][key].to(configs.device).long())

        start_positions[0] = torch.autograd.Variable(start_positions[0].to(configs.device).long())
        end_positions[0] = torch.autograd.Variable(end_positions[0].to(configs.device).long())
        start_positions[1] = torch.autograd.Variable(start_positions[1].to(configs.device).long())
        end_positions[1] = torch.autograd.Variable(end_positions[1].to(configs.device).long())
        start_positions[2] = torch.autograd.Variable(start_positions[2].to(configs.device).long())
        end_positions[2] = torch.autograd.Variable(end_positions[2].to(configs.device).long())
        return question_context_features, start_positions, end_positions

    def siamese_encoder(self, question_context_features):
        encoding_domain_0 = self.encoder(question_context_features[0])
        encoding_domain_1 = self.encoder(question_context_features[1])
        encoding_domain_2 = self.encoder(question_context_features[2])
        return [encoding_domain_0, encoding_domain_1, encoding_domain_2]

    def siamese_discriminator(self, encodings, start_positions, end_positions):
        encoding_domain_0, encoding_domain_1, encoding_domain_2 = encodings

        reversed_representation_0 = self.gradient_reversal_layer(encoding_domain_0)
        reversed_representation_0, aux_qa_outputs_0 = self.disc_aux_qa_layer(reversed_representation_0,
                                                                             start_positions[0],
                                                                             end_positions[0])

        reversed_representation_1 = self.gradient_reversal_layer(encoding_domain_1)
        reversed_representation_1, _ = self.disc_aux_qa_layer(reversed_representation_1,
                                                                             start_positions[1],
                                                                             end_positions[1])

        reversed_representation_2 = self.gradient_reversal_layer(encoding_domain_2)
        reversed_representation_2, _ = self.disc_aux_qa_layer(reversed_representation_2,
                                                                             start_positions[2],
                                                                             end_positions[2])
        
        triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y),
            margin=2.0, swap=True, reduction='none')

        adv_loss = triplet_loss(anchor=reversed_representation_0,
                                positive=reversed_representation_1,
                                negative=reversed_representation_2)[0].cuda()
        
        aux_qa_loss = aux_qa_outputs_0.loss #*0.5

        return adv_loss, \
               aux_qa_outputs_0, \
               aux_qa_loss

    def gradient_reversal_layer(self, encoding):
        reversed_feature = ReverseLayerF.apply(encoding.last_hidden_state,
                                               configs.reverse_layer_lambda)  #ref: https://github.com/SongweiGe/scDGN/blob/master/utils/model_util.py
        return reversed_feature

    def siamese_factoid_qa_out_gen(self, encodings, start_positions, end_positions):
        encoding_domain_0, encoding_domain_1, encoding_domain_2 = encodings

        factoid_qa_output_0 = self.factoid_qa_output_generator(encoding_domain_0,
                                                               start_positions[0],
                                                               end_positions[0])
        total_loss = factoid_qa_output_0.loss #*0.5
        return factoid_qa_output_0, total_loss