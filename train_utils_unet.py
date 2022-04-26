import logging
import pathlib

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_model(criterion_class,  model, optimizer, trainloader, summary_writer, epoch, clip=100.0):

    running_loss_mse = 0
    running_loss_l1 = 0
    running_loss_cls = 0
    running_accuracy = 0

    model.train()
    for clear, noise, labels in tqdm(trainloader):
        noise, clear = noise.cuda(non_blocking=True).unsqueeze(1), clear.cuda(non_blocking=True).unsqueeze(1)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        prclr = model.forward(noise)

        loss_l1 = criterion_class[0](prclr[0], clear)
        losmse = criterion_class[1](prclr[0], clear)
        loss_cls = criterion_class[2](prclr[1], labels)


        loss = losmse + 0.15 * loss_cls

        loss.backward()
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        running_loss_mse += losmse.item()
        running_loss_l1 += loss_l1.item()
        running_loss_cls += loss_cls.item()

        optimizer.step()

        ps = torch.exp(prclr[1])
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


    train_loss_mse = running_loss_mse/(len(trainloader))
    train_loss_l1 = running_loss_l1/(len(trainloader))
    train_loss_cls = running_loss_cls/(len(trainloader))
    train_loss_acc = running_accuracy/(len(trainloader))

    summary_writer.add_scalar("train_mse", train_loss_mse, epoch + 1)
    # summary_writer.add_scalar("train_l1", train_loss_l1, epoch + 1)
    summary_writer.add_scalar("train_cls", train_loss_cls, epoch + 1)
    summary_writer.add_scalar("train_loss", 0.15 * train_loss_cls + train_loss_mse, epoch + 1)
    summary_writer.add_scalar("train_acc", train_loss_acc, epoch + 1)





def eval_model(criterion_class,  model, testloader, summary_writer, epoch, dirs, weight_matrix=None):
    model_dir, conf_dir = dirs

    running_loss_mse = 0
    running_loss_l1 = 0
    running_loss_cls = 0
    running_accuracy = 0

    model.eval()
    with torch.no_grad():
        for clear, noise, labels in tqdm(testloader):
            noise, clear = noise.cuda(non_blocking=True).unsqueeze(1), clear.cuda(non_blocking=True).unsqueeze(1)
            labels = labels.cuda(non_blocking=True)

            prclr = model.forward(noise)

            # losmse = criterion_class(prclr, clear)

            loss_l1 = criterion_class[0](prclr[0], clear)
            losmse = criterion_class[1](prclr[0][labels==1], clear[labels==1])
            loss_cls = criterion_class[2](prclr[1], labels)
            # loss = losmse + loss_cls

            running_loss_mse += losmse.item()
            running_loss_l1 += loss_l1.item()
            running_loss_cls += loss_cls.item()

            ps = torch.exp(prclr[1])
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss_mse = running_loss_mse/(len(testloader))
    test_loss_l1 = running_loss_l1/(len(testloader))
    test_loss_cls = running_loss_cls/(len(testloader))
    test_loss_acc = running_accuracy/(len(testloader))

    _save_eval_results(conf_dir, epoch, model, model_dir, summary_writer, test_loss_mse)
    # summary_writer.add_scalar("test_l1", test_loss_l1, epoch + 1)
    summary_writer.add_scalar("test_cls", test_loss_cls, epoch + 1)
    summary_writer.add_scalar("test_loss", 0.15*test_loss_cls + test_loss_mse, epoch + 1)
    summary_writer.add_scalar("test_acc", test_loss_acc, epoch + 1)



def _save_eval_results(conf_dir, epoch, model: torch.nn.Module,
                       model_dir, summary_writer, test_mse,):

    summary_writer.add_scalar("test_mse", test_mse, epoch + 1)

    torch.save(model.state_dict(), f'{model_dir}/{int(test_mse * 1000)}_{epoch}_state_dict.pth')




def create_meta_dirs(target_name, arch, batch_size, epochs, freeze, iteration, len_trainloader, lr, work_dir):
    postfix = f"{arch}/{'fr' if freeze else 'unfr'}/" \
        f"it={iteration}/ep={epochs} bs={batch_size} lr={lr} len_train={len_trainloader}"

    log_dir = work_dir + f'results/{target_name}/logs/'
    conf_dir = work_dir + f'results/{target_name}/conf/'
    model_dir = work_dir + f'results/{target_name}/model/'

    log_dir += postfix
    conf_dir += postfix
    model_dir += postfix

    pathlib.Path(conf_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    return conf_dir, log_dir, model_dir


def update_scheduler(epoch, scheduler, summary_writer):
    scheduler.step()
    summary_writer.add_scalar("lr", scheduler.get_lr()[0], epoch + 1)
