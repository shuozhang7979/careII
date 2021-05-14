from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.k_folder import DataList
import torch


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # get training options
    data_list = DataList(opt.dataset_root)

    val_list = data_list.get_current_list(phase='val', k=opt.k_fold, time=opt.time)
    val_dataset = create_dataset(opt, val_list).__iter__()

    train_list = data_list.get_current_list(phase='train', k=opt.k_fold, time=opt.time)
    train_dataset = create_dataset(opt, train_list)

    model = create_model(opt)      # create a model given opt. model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt, model)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        # outer loop for different epochs;

        for i, data in enumerate(train_dataset):  # inner loop within one epoch

            total_iters += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.interval_print_train_loss == 0:
                visualizer.print_current_losses('train', model.get_current_losses(), total_iters)
                # model.histogram_vis(visualizer.summary, total_iters)

            if total_iters % opt.interval_print_train_img == 0:
                model.compute_visuals()
                visualizer.print_current_img('train', model.get_current_visuals(), total_iters)

            if total_iters % opt.interval_val == 0:
                val_data = next(val_dataset)
                model.set_input(val_data)
                model.test()
                visualizer.print_current_losses('val', model.get_current_losses(), total_iters)
                visualizer.print_current_img('val', model.get_current_visuals(), total_iters)

            # if total_iters % 1000 == 0:
            #     model.histogram_vis(visualizer.summary, total_iters)

        if total_iters % opt.interval_save_model == 0:
            model.save_networks(epoch)

        model.update_learning_rate_(visualizer.summary, total_iters, 0)
        # update learning rates in the beginning of every epoch.
