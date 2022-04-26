import json
import logging
import pathlib

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_model(criterion_class, model, optimizer, trainloader, summary_writer, epoch, clip=100.0):
    running_accuracy = 0
    running_loss = 0

    model.train()
    for X, y in tqdm(trainloader):
        X, y = X.cuda(non_blocking=True).unsqueeze(1), y.cuda(non_blocking=True)

        optimizer.zero_grad()
        logcl = model.forward(X)

        losspos = criterion_class(logcl, y)

        loss = losspos

        loss.backward()
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        ps = torch.exp(logcl)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)

        running_loss += losspos.item()

        running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = running_accuracy / len(trainloader)

    summary_writer.add_scalar("train_loss", train_loss, epoch + 1)
    summary_writer.add_scalar("train_accuracy", train_accuracy, epoch + 1)



def eval_model(criterion_class, model, testloader, summary_writer, epoch, dirs):
    model_dir, conf_dir = dirs

    running_loss = 0
    running_accuracy = 0
    y_pred = np.array([])
    y_true = np.array([])

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(testloader):
            X, y = X.cuda(non_blocking=True).unsqueeze(1), y.cuda(non_blocking=True)

            logcl = model.forward(X)

            losspos = criterion_class(logcl, y)

            ps = torch.exp(logcl)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y.view(*top_class.shape)

            running_loss += losspos.item()
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            y_pred = np.append(y_pred, top_class.cpu().numpy())
            y_true = np.append(y_true, y.view(*top_class.shape).cpu().numpy())

    test_accuracy = running_accuracy / len(testloader)
    test_loss = running_loss / len(testloader)

    _save_eval_results(conf_dir, epoch, model, model_dir, summary_writer, test_accuracy, test_loss, y_pred, y_true)


def _save_eval_results(conf_dir, epoch, model: torch.nn.Module,
                       model_dir, summary_writer, test_accuracy, test_loss, y_pred, y_true):
    summary_writer.add_scalar("test_loss", test_loss, epoch + 1)
    summary_writer.add_scalar("test_accuracy", test_accuracy, epoch + 1)

    torch.save(model.state_dict(), f'{model_dir}/{int(test_accuracy * 1000)}_{epoch}_state_dict.pth')

    with open(f'{conf_dir}/{int(test_accuracy * 1000)}_{epoch}.json', 'w') as f:
        json.dump({'y_pred': y_pred.tolist(), 'y_true': y_true.tolist()}, f)


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
