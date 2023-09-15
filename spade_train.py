import jittor as jt
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from pix2pix_trainer import Pix2PixTrainer
import warnings

warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
if jt.rank == 0:
    visualizer = Visualizer(opt)

# inp = jt.Var(np.random.rand(4, 1, 384, 512))
# visualizer.plot_model(trainer.pix2pix_model, inp)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        # print(data_i['image'])
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            if jt.rank == 0:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            if jt.rank == 0:
                visuals = OrderedDict([('input_label', data_i['label']),
                                       ('synthesized_image', trainer.get_latest_generated()),
                                       ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            if jt.rank == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()


    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        if jt.rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save(epoch)
            trainer.save('latest')


print('Training was successfully finished.')
