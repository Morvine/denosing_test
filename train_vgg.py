import logging

from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim



import dataloaderVGG
from model_vgg import VGG
from train_utils_vgg import train_model, eval_model,  update_scheduler, \
    create_meta_dirs

logger = logging.getLogger(__name__)


def train(data_dir,
          data_fir_val,
          epochs=1_000,
          iteration='',
          lr=1e-4,
          gamma=0.1,
          batch_size=32,
          work_dir='../',
          scheduler_step_size=10,
          scheduler_milestones=(),
          weight_decay=1e-5,
          debug=False,
          main_meta_dir='',
          clip=100.0,
          ):
    logger.setLevel(logging.DEBUG if debug else logging.ERROR)
    dataloaderVGG.logger.setLevel(logging.DEBUG if debug else logging.ERROR)

    trainloader, testloader = dataloaderVGG.create_dataloaders(data_dir,
                                                                 data_fir_val, batch_size)

    n_classes, len_trainloader = 2, len(trainloader.dataset)

    conf_dir, log_dir, model_dir = create_meta_dirs(main_meta_dir, 'vgg',
                                                    batch_size, epochs, False,
                                                    iteration, len_trainloader, lr, work_dir)
    model = VGG()

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    model.cuda()


    criterion_class = nn.CrossEntropyLoss()

    if scheduler_milestones:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(scheduler_milestones), gamma=gamma)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    logger.debug(f"train: {iteration, lr, epochs, data_dir, log_dir}")
    with SummaryWriter(log_dir=log_dir) as summary_writer:
        for epoch in range(0, epochs):
            logger.debug(f"\ntrain: epoch {epoch + 1}/{epochs}\n")

            train_model(criterion_class,  model, optimizer, trainloader, summary_writer, epoch, clip)
            eval_model(criterion_class, model, testloader, summary_writer, epoch, [model_dir, conf_dir])
            # show_predictions(model, evalloader, epoch, summary_writer)
            update_scheduler(epoch, scheduler, summary_writer)


if __name__ == "__main__":
    datadir = "D:\\dataset\\zwuk\\train"
    data_fir_val = "D:\\dataset\\zwuk\\val"
    epoch = 20
    lr = 1e-3
    model_depth = 10
    scheduler_step_size = 3
    weight_decay = 1e-4

    iteration = 'n1_vgg'
    work_dir = '../'

    model_complexity = 45

    gamma = 0.8

    batch_size = 50

    train(datadir,
          data_fir_val,
          epochs=epoch,
          iteration=iteration,
          lr=lr,
          gamma=gamma,
          batch_size=batch_size,
          work_dir='../',
          scheduler_step_size=scheduler_step_size,
          scheduler_milestones=(),
          weight_decay=weight_decay,
          debug=False,
          main_meta_dir='',
          clip=50.0,
          )