from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--name', type=str,
                            default='myDice',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--info', type=str,
                            default='input normalize into 01')
        parser.add_argument('--model', type=str, default='multiclass_dice',
                            help='chooses which model to use.'
                                 'unet | unet_gan | multiclass_ce | multiclass_dice')

        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

        parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        parser.add_argument('--interval_print_train_loss', type=int, default=8*100,
                            help='number of iters')
        parser.add_argument('--interval_print_train_img', type=int, default=8*400,
                            help='number of iters')
        parser.add_argument('--interval_val', type=int, default=8*400,
                            help='number of iters')
        parser.add_argument('--interval_save_model', type=int, default=3,
                            help='number of epochs')
        parser.add_argument('--k_fold', type=int, default=9,
                            help='number of epochs')
        parser.add_argument('--time', type=int, default=0,
                            help='[0,k-2)')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--netG', type=str, default='unet_512',
                            help='')
        parser.add_argument('--lambda_class', type=float, default=0.0, help='weight for class loss')
        parser.add_argument('--lambda_dice', type=float, default=1.0, help='weight for dice loss')

        self.isTrain = True
        return parser
