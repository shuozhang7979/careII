import torch
from .base_model import BaseModel
from . import networks
import numpy
import matplotlib.pyplot as plt


class MulticlassDiceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        :param is_train: -- original option parser
        :param parser:   -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:         the modified parser.
        """
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.visual_names = ['one_slice_v', 'masks_v', 'fake_masks_v', 'fake_masks_v_diff',
                             'fake_masks_v_ical_outer', 'fake_masks_v_icar_outer',
                             'fake_masks_v_ecal_outer', 'fake_masks_v_ecar_outer',
                             ]
        self.one_slice_v = None
        self.masks_v = None
        self.fake_masks_v = None
        self.fake_masks_v_diff = None
        self.fake_masks_v_ical_outer = None
        self.fake_masks_v_icar_outer = None
        self.fake_masks_v_ecal_outer = None
        self.fake_masks_v_ecar_outer = None

        self.loss_names = ['dice']    # loss_G_L1
        self.model_names = ['G']    # netG
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_num_channels, opt.output_num_channels, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.slices = None
        self.masks = None
        self.labels = None
        self.effective_channels = []
        self.fake_masks = None
        self.fake_label_masks = None
        self.back_ground = None

        if self.isTrain:
            # define loss functions
            self.loss_all = None
            # self.class_loss = torch.nn.CrossEntropyLoss().to(self.device)
            # self.class_loss = lovasz_losses.lovasz_softmax()
            self.dice_loss = networks.MySoftDiceLoss().to(self.device)
            # self.dice_loss = networks.SoftDiceLoss().to(self.device)
            self.loss_dice = None
            self.loss_class = None
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input_):

        self.slices, self.masks, self.labels, inf = input_[0],  input_[1].to(self.device), input_[2], input_[3]

        self.effective_channels = []
        for i, str_i in enumerate(inf):
            self.effective_channels.append([])
            for j in str_i:
                self.effective_channels[i].append(int(j))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.one_slice = self.one_slice.double()
        self.fake_masks = self.netG(self.slices)
        # self.fake_label_masks = masks_to_labels(self.fake_masks)

    def backward_g(self):
        """Calculate L1 loss for the generator"""
        # self.loss_class = self.class_loss(self.fake_label_masks, self.labels)
        # self.loss_class = lovasz_losses.lovasz_softmax(self.fake_masks, self.labels)
        self.loss_dice = self.dice_loss(self.fake_masks, self.masks, self.effective_channels)
        # combine loss and calculate gradients
        self.loss_all = self.opt.lambda_dice * self.loss_dice   # + self.opt.lambda_class * self.loss_class
        self.loss_all.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero

        self.backward_g()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
        torch.cuda.empty_cache()

    def test(self):
        self.netG.eval()
        self.fake_masks = self.netG(self.slices)
        # self.loss_class = self.class_loss(self.fake_label_masks, self.labels)
        # self.loss_class = lovasz_losses.lovasz_softmax(self.fake_label_masks, self.labels)
        self.loss_dice = self.dice_loss(self.fake_masks, self.masks, self.effective_channels)
        self.compute_visuals()
        torch.cuda.empty_cache()

    def compute_visuals(self):
        key = numpy.random.randint(self.opt.batch_size)
        self.one_slice_v = self.slices[key, ...].detach().cpu()
        self.masks_v = torch.zeros([512, 512])
        self.fake_masks_v = torch.zeros([512, 512])
        for i in range(self.masks.shape[1]):
            self.masks_v += self.masks[key, i, ...].detach().cpu()
            self.fake_masks_v += self.fake_masks[key, i, ...].detach().cpu()
        self.masks_v = self.masks_v.unsqueeze(0)
        self.fake_masks_v = self.fake_masks_v.unsqueeze(0)
        self.fake_masks_v_diff = torch.abs(self.masks_v - self.fake_masks_v)
        self.fake_masks_v_ical_outer = self.fake_masks[key, 0, ...].unsqueeze(0)
        self.fake_masks_v_icar_outer = self.fake_masks[key, 1, ...].unsqueeze(0)
        self.fake_masks_v_ecal_outer = self.fake_masks[key, 2, ...].unsqueeze(0)
        self.fake_masks_v_ecar_outer = self.fake_masks[key, 3, ...].unsqueeze(0)

    def add_net_graph(self, summary):
        init_input = torch.zeros((1, 1, 512, 512), device=self.device)
        summary.add_graph(self.netG, init_input)

    def update_learning_rate_(self, summary, itr, metric=0):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # old_lr = self.optimizers[0].param_groups[0]['lr']
        super().update_learning_rate(metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        # self.opt.lr = self.opt.lr * 0.8
        # adjust_learning_rate(self.optimizer_G, self.opt.lr)
        summary.add_scalar('LearningRate', lr, itr)

    def histogram_vis(self, summary, itr):
        for name, param in self.netG.named_parameters():
            if name == 'module.model.model.1.model.3.model.3.model.5.weight':
                pass
                # print(param.grad)
                # summary.add_histogram(name + str(itr), param.clone().cpu().data.numpy())


# def adjust_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         print("current lr is ", param_group["lr"])
#         if param_group["lr"] > lr:
#             param_group["lr"] = lr
#
# #
# def masks_to_labels(masks):
#     labels = torch.zeros([masks.shape[0], 9, masks.shape[2], masks.shape[3]]).cuda()
#     back_ground = torch.ones([masks.shape[0], masks.shape[2], masks.shape[3]]).cuda()
#     for i in range(0, 4):
#         labels[:, 2*i+2, ...] = (masks[:, 2*i+1, ...] - masks[:, 2*i, ...])
#         labels[:, 2*i+1, ...] = masks[:, 2*i, ...]
#         back_ground -= masks[:, 2*i+1, ...]
#     labels[:, 0, ...] = back_ground
#     return labels

