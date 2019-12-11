import os

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from zs3.dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(os.path.join(self.directory))
        return writer

    def visualize_image(
        self,
        writer,
        dataset,
        image,
        target,
        output,
        global_step,
        name="Train",
        nb_image=3,
    ):
        grid_image = make_grid(
            image[:nb_image].clone().cpu().data, nb_image, normalize=True
        )
        writer.add_image(name + "_Image", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.max(output[:nb_image], 1)[1].detach().cpu().numpy(),
                dataset=dataset,
            ),
            nb_image,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image(name + "_Predicted label", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.squeeze(target[:nb_image], 1).detach().cpu().numpy(),
                dataset=dataset,
            ),
            nb_image,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image(name + "_Groundtruth label", grid_image, global_step)

    def visualize_image_validation(
        self,
        writer,
        dataset,
        image,
        target,
        output,
        global_step,
        name="Train",
        nb_image=3,
    ):
        grid_image = make_grid(image.data, nb_image, normalize=True)
        writer.add_image(name + "_Image", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.max(output, 1)[1].detach().numpy(), dataset=dataset
            ),
            nb_image,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image(name + "_Predicted label", grid_image, global_step)
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.squeeze(target[:nb_image], 1).detach().numpy(), dataset=dataset
            ),
            nb_image,
            normalize=False,
            range=(0, 255),
        )
        writer.add_image(name + "_Groundtruth label", grid_image, global_step)
