import torch
import torch.nn as nn
import torch.nn.functional as F


class PreTrainedVT(nn.Module):
    def __init__(self):
        super(PreTrainedVT, self).__init__()
        self.image_size = 300

        # same layers as VisualTransformer visual representation learning part
        self.global_conv = nn.Conv2d(512, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)

        self.local_embedding = nn.Sequential(
            nn.Linear(256, 255),
            nn.ReLU(),
        )

        self.visual_transformer = nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            dropout=0,
            activation="relu",
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0),
        )

        # pretraining network action predictor, should be used in Visual Transformer model
        self.pretrain_fc = nn.Linear(3136, 6)


    def forward(self, global_feature: torch.Tensor, local_feature: dict):
        batch_size = global_feature.shape[0]

        global_feature = global_feature.squeeze(dim=1)
        image_embedding = F.relu(self.global_conv(global_feature))
        image_embedding = image_embedding + self.global_pos_embedding.repeat([batch_size, 1, 1, 1]).cuda()
        image_embedding = image_embedding.reshape(batch_size, -1, 49)

        if self.wo_location_enhancement:
            detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
            local_input = torch.cat((detection_input_features, local_feature['indicator']), dim=2)
        else:
            detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
            local_input = torch.cat((
                detection_input_features,
                local_feature['labels'].unsqueeze(dim=2),
                local_feature['bboxes'] / self.image_size,
                local_feature['scores'].unsqueeze(dim=2),
                local_feature['indicator']
            ), dim=2)

        visual_representation, _ = self.visual_transformer(src=local_input, query_embed=image_embedding)

        visual_rep = self.visual_rep_embedding(visual_representation)
        visual_rep = visual_rep.reshape(batch_size, -1)

        action = self.pretrain_fc(visual_rep)

        return {
            'action': action,
            'fc_weights': self.pretrain_fc.weight,
            'visual_reps': visual_rep.reshape(batch_size, 64, 49)
        }


def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos
