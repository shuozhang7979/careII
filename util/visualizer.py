from tensorboardX import SummaryWriter
import torch
import math
import numpy


class Visualizer:
    """
    This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'tensorboard' for display,
    and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, model):
        """Initialize the Visualizer class
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.summary = SummaryWriter(logdir='./checkpoints/%s/logs/' % self.name)
        # add network graph into tensorboard
        model.add_net_graph(self.summary)

    def print_current_losses(self, phase, current_losses, total_iters):
        tags = list(current_losses.keys())
        for tag in tags:
            self.summary.add_scalar(phase + tag, current_losses[tag], total_iters)
            print('%s, %s, %s, %d' % (phase, tag, current_losses[tag], total_iters))

    def print_current_img(self, phase, current_imgs, total_iters):
        imgs = list(current_imgs.keys())
        row = 4
        lines = math.ceil(len(imgs) / row)
        res = torch.zeros([1, 512 * lines, 512 * row])
        for i, img in enumerate(imgs):
            res[0, 512 * (i//row): 512 * (i//row + 1), 512 * (i % row):512 * (i % row + 1)]\
                = current_imgs[img].cpu()
        self.summary.add_image(phase + str(total_iters), res)

    def print_confusion_matrix(self, phase, masks, fake_masks, total_iters):
        threshold = 10.0

        def mask_array(mask):
            mask_np = numpy.zeros([5])
            for i in range(4):
                if torch.sum(mask[2 * i, ...]) > threshold:
                    mask_np[i] = 1
            if mask_np.sum() == 0:
                mask_np[4] = 1
            return mask_np
        CM = numpy.zeros([5, 5])
        masks_np = mask_array(masks)
        fake_masks_np = mask_array(fake_masks)
        self.summary.add_figure()

