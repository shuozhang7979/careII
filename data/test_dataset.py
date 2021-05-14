from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.k_folder import DataList


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    data_list = DataList(opt.dataset_root)

    val_list = data_list.get_current_list(phase='val', k=opt.k_fold, time=opt.time)
    val_dataset = create_dataset(opt, val_list)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt, model)   # create a visualizer that display/save images and plots

    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        # outer loop for different epochs;
        train_list = data_list.get_current_list(phase='train', k=opt.k_fold, time=opt.time)
        train_dataset = create_dataset(opt, train_list)  # create a dataset given opt.dataset_mode and other options
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            d = data[0]
            d.cuda()
            print(d[1])

