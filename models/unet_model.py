# import torch
# from .base_model import BaseModel
# from . import networks
#
#
# class UNetModel(BaseModel):
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.
#         :param is_train: -- original option parser
#         :param parser:   -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
#         Returns:         the modified parser.
#         """
#         parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='careiichallenge')
#         if is_train:
#             # parser.set_defaults(pool_size=0, gan_mode='vanilla')
#             parser.add_argument('--lambda_L1', type=float, default=2.0, help='weight for L1 loss')
#             parser.add_argument('--lambda_dice', type=float, default=1.0, help='weight for dice loss')
#
#         return parser
#
#     def __init__(self, opt):
#         """Initialize the pix2pix class.
#
#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseModel.__init__(self, opt)
#         self.fake_masks_0 = None
#         self.fake_masks_1 = None
#         self.real_masks_0 = None
#         self.real_masks_1 = None
#         self.visual_names = ['fake_masks_0', 'real_masks_0', 'fake_masks_1',  'real_masks_1']
#         self.loss_names = ['G_L1', 'G_dice']    # loss_G_L1
#         self.model_names = ['G']    # netG
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#         # define networks (both generator and discriminator)
#         self.netG = networks.defin_G(opt.input_num_channels, opt.output_num_channels, opt.ngf, opt.netG, opt.norm,
#                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.one_slice = None
#         self.masks = None
#         self.fake_masks = None
#
#         if self.isTrain:
#             # define loss functions
#             self.loss_G = None
#             self.criterionL1 = torch.nn.L1Loss().to(self.device)
#             self.loss_G_L1 = None
#             self.diceLoss = networks.SoftDiceLoss().to(self.device)
#             self.loss_G_dice = None
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#
#     def set_input(self, input_):
#         self.one_slice, self.masks, _ = input_
#
#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         # self.one_slice = self.one_slice.double()
#         self.fake_masks = self.netG(self.one_slice)  # G(A)
#
#     def backward_G(self):
#         """Calculate L1 loss for the generator"""
#         self.loss_G_L1 = self.criterionL1(self.masks, self.fake_masks)
#         self.loss_G_dice = self.diceLoss(self.masks, self.fake_masks)
#         # combine loss and calculate gradients
#         self.loss_G = self.opt.lambda_L1 * self.loss_G_L1 + self.opt.lambda_dice * self.loss_G_dice
#         self.loss_G.backward()
#
#     def optimize_parameters(self):
#         self.forward()  # compute fake images: G(A)
#         # update G
#         self.optimizer_G.zero_grad()  # set G's gradients to zero
#         self.backward_G()  # calculate gradients for G
#         self.optimizer_G.step()  # update G's weights
#
#     def compute_visuals(self):
#         self.fake_masks_0 = self.fake_masks[0, 0, :, :].unsqueeze(0)
#         self.fake_masks_1 = self.fake_masks[0, 1, :, :].unsqueeze(0)
#         self.real_masks_0 = self.masks[0, 0, :, :].unsqueeze(0)
#         self.real_masks_1 = self.masks[0, 1, :, :].unsqueeze(0)
#
#     def add_net_graph(self, summary):
#         init_input = torch.zeros((1, 1, 512, 512), device=self.device)
#         summary.add_graph(self.netG, init_input)
