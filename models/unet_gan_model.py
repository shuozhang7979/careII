# import torch
# from .base_model import BaseModel
# from . import networks
# import numpy
#
#
# class UNetGANModel(BaseModel):
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.
#         :param is_train: -- original option parser
#         :param parser:   -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
#         Returns:         the modified parser.
#         """
#         parser.set_defaults(norm='batch', netG='unet_256', netD='basic', n_layers_D=3)
#         if is_train:
#             # parser.set_defaults(pool_size=0, gan_mode='vanilla')
#             parser.add_argument('--lambda_L1', type=float, default=0.0, help='weight for L1 loss')
#             parser.add_argument('--lambda_dice', type=float, default=2.0, help='weight for dice loss')
#             parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for dice loss')
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
#         self.show_slice = None
#         self.fake_masks = None
#         # self.total_masks = None
#         # self.total_mask_0 = None
#         # self.total_mask_1 = None
#         self.visual_names = ['show_slice',
#                              # 'total_mask_0', 'total_mask_1',
#                              'real_masks_0', 'fake_masks_0', 'real_masks_1', 'fake_masks_1']
#         self.loss_names = ['G_L1', 'G_DICE', 'G_GAN', 'D']  # loss_G_L1
#         self.model_names = ['G', 'D']  # netG
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#         # define networks (both generator and discriminator)
#         self.netG = networks.define_G(opt.input_num_channels, opt.output_num_channels, opt.ngf, opt.netG, opt.norm,
#                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.netD = networks.define_D(opt.input_num_channels + opt.output_num_channels, opt.ndf, opt.netD,
#                                       opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.one_slice = None
#         self.masks = None
#
#         if self.isTrain:
#             # define loss functions
#             self.loss_G = None
#             self.criterionL1 = torch.nn.L1Loss().to(self.device)
#             self.loss_G_L1 = None
#             self.diceLoss = networks.SoftDiceLoss().to(self.device)
#             self.loss_G_DICE = None
#             self.ganloss = networks.GANLoss('lsgan').to(self.device)
#             self.loss_G_GAN = None
#             self.loss_D_fake = None
#             self.loss_D = None
#             self.loss_D_real = None
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)
#
#     def set_input(self, input_):
#         # self.one_slice, self.masks, self.total_masks, _ = input_
#         self.one_slice, self.masks, _ = input_
#
#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         # self.one_slice = self.one_slice.double()
#         self.fake_masks = self.netG(self.one_slice)  # G(A)
#
#     def backward_G(self):
#         """Calculate L1 loss for the generator"""
#         fake_pair = torch.cat((self.one_slice, self.masks), 1)
#         pred_fake = self.netD(fake_pair)
#         self.loss_G_GAN = self.ganloss(pred_fake, True)
#         self.loss_G_L1 = self.criterionL1(self.masks, self.fake_masks)
#         self.loss_G_DICE = self.diceLoss(self.masks, self.fake_masks)
#         # combine loss and calculate gradients
#         self.loss_G = self.opt.lambda_L1 * self.loss_G_L1 + \
#                       self.opt.lambda_dice * self.loss_G_DICE + \
#                       self.opt.lambda_gan * self.loss_G_GAN
#         self.loss_G.backward()
#
#     def backward_D(self):
#         """Calculate GAN loss for the discriminator"""
#         # Fake; stop backprop to the generator by detaching fake_B
#         fake_pair = torch.cat((self.one_slice, self.fake_masks),
#                               1)  # we use conditional GANs; we need to feed both input and output to the discriminator
#         pred_fake = self.netD(fake_pair)
#         self.loss_D_fake = self.ganloss(pred_fake.detach(), False)
#         # Real
#         real_pair = torch.cat((self.one_slice, self.masks), 1)
#         pred_real = self.netD(real_pair)
#         self.loss_D_real = self.ganloss(pred_real, True)
#         # combine loss and calculate gradients
#         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
#         self.loss_D.backward()
#
#     def optimize_parameters(self):
#         self.forward()  # compute fake images: G(A)
#         # update D
#         self.set_requires_grad(self.netD, True)  # enable backprop for D
#         self.optimizer_D.zero_grad()  # set D's gradients to zero
#         self.backward_D()  # calculate gradients for D
#         self.optimizer_D.step()  # update D's weights
#         # update G
#         self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
#         self.optimizer_G.zero_grad()  # set G's gradients to zero
#         self.backward_G()  # calculate graidents for G
#         self.optimizer_G.step()  # udpate G's weights
#
#     def compute_visuals(self):
#         key = numpy.random.randint(self.opt.batch_size)
#         self.fake_masks_0 = self.fake_masks[key, 0, :, :].unsqueeze(0)
#         self.fake_masks_1 = self.fake_masks[key, 1, :, :].unsqueeze(0)
#         self.real_masks_0 = self.masks[key, 0, :, :].unsqueeze(0)
#         self.real_masks_1 = self.masks[key, 1, :, :].unsqueeze(0)
#         self.show_slice = self.one_slice[key, ...]
#         # self.total_mask_0 = self.total_masks[key, 0, :, :].unsqueeze(0)
#         # self.total_mask_1 = self.total_masks[key, 1, :, :].unsqueeze(0)
#
#     def add_net_graph(self, summary):  # vistualizer
#         init_input = torch.zeros((1, 1, 512, 512), device=self.device)
#         summary.add_graph(self.netG, init_input)
