import json
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchbearer
from torchbearer import Trial, Callback
from torchbearer.callbacks import TensorBoard, TensorBoardText, MultiStepLR, \
    TorchScheduler

FORCE_MPS = False


def set_seed(seed):
    """
    Set all the seeds for reproducibility.

    :param seed: desired seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def get_device(device):
    """
    Get the a device to use; supports 'auto' for best guess

    :param device: the desired device
    :return: the actual device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps' if FORCE_MPS else 'cpu'
        else:
            device = 'cpu'

    return torch.device(device)


def fit_model(model, criterion, opt, trainloader, valloader, epochs=1000,
              schedule=None, gamma=None, run_id=None, log_dir=None,
              model_file=None, resume=None, device='auto', verbose=0,
              pre_extra_callbacks=None, extra_callbacks=None,
              acc='acc', period=None, args=None):
    """
    Train a model, logging and snapshotting along the way

    :param model: the model to train
    :param criterion: loss function
    :param opt: optimiser
    :param trainloader: training data loader
    :param valloader: validation data loader
    :param epochs: number of epochs
    :param schedule: a list of learning rate drop points (e.g. [100, 150] or
    None) or an instance of TorchScheduler
    :param gamma: amount to drop learning rate by at each point if schedule
    is not None or a TorchScheduler instance
    :param run_id: identifier of the run; used to determine the file name of
    logs
    :param log_dir: location to save log files
    :param model_file: name of model file to save
    :param resume: starts from a previously saved model
    :param device: compute device
    :param verbose: Verbosity level. 0=No printing, 1=progress bar for entire
    training, 2=progress bar for one epoch.
    :param pre_extra_callbacks: : extra callbacks to prepend to training loop
    :param extra_callbacks: extra callbacks to add to training loop
    :param acc: override the accuracy measurement (default 'acc' set based on
    loss)
    :param period: how often to save the model checkpoints (can be None)
    :param args: argsparse arguments or params dict to be logged
    :return:
    """
    print('==> Setting up callbacks..')

    device = get_device(device)

    if not model_file.endswith('.pt'):
        model_file += '.pt'

    model_file_checkpoint = model_file.replace(".pt", ".{epoch:03d}.pt")

    cb = []
    if log_dir is not None:
        # TODO: saving init state (optional)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S') + "-run-" + str(
            run_id)
        tboard = TensorBoard(write_graph=False, comment=current_time,
                             log_dir=log_dir)
        tboardtext = TensorBoardText(write_epoch_metrics=False,
                                     comment=current_time, log_dir=log_dir)

        @torchbearer.callbacks.on_start
        def write_params(_):
            params = {'model': str(model), 'criterion': str(criterion),
                      'opt': str(opt), 'trainloader': str(trainloader),
                      'valloader': str(valloader), 'schedule': str(schedule),
                      'run_id': str(run_id),
                      'log_dir': str(log_dir), 'model_file': str(model_file),
                      'resume': str(resume),
                      'device': str(device)}
            df = pd.DataFrame(params, index=[0]).transpose()
            tboardtext.get_writer(tboardtext.log_dir).add_text('params',
                                                               df.to_html(), 1)
            df.to_json(tboardtext.log_dir + '-params.json')

            if args is not None:
                myargs = args if isinstance(args, dict) else vars(args)
                tboardtext.get_writer(tboardtext.log_dir).add_text('args',
                                                                   str(myargs),
                                                                   1)
                with open(tboardtext.log_dir + '-args.json', 'w') as fp:
                    json.dump(myargs, fp)

        cb.extend([tboard, tboardtext, write_params])

    if extra_callbacks is not None:
        if not isinstance(extra_callbacks, (list, tuple)):
            extra_callbacks = [extra_callbacks]
        cb.extend(extra_callbacks)

    if pre_extra_callbacks is not None:
        if not isinstance(pre_extra_callbacks, (list, tuple)):
            pre_extra_callbacks = [pre_extra_callbacks]
        cb = pre_extra_callbacks + cb

    if model_file is not None:
        cb.append(torchbearer.callbacks.MostRecent(
            model_file.replace(".pt", "_last.pt")))
        if period is not None:
            cb.append(torchbearer.callbacks.Interval(model_file_checkpoint,
                                                     period=period,
                                                     on_batch=False))
    if schedule is not None:
        if isinstance(schedule, TorchScheduler):
            cb.append(schedule)
        else:
            cb.append(MultiStepLR(schedule, gamma=gamma))

    print('==> Training model..')
    print('using device: ' + str(device))
    metrics = ['loss', 'lr']
    if acc is not None:
        if not isinstance(acc, (list, tuple)):
            metrics.append(acc)
        else:
            metrics.extend(acc)
    trial = Trial(model, opt, criterion, metrics=metrics, callbacks=cb)
    trial.with_generators(train_generator=trainloader,
                          val_generator=valloader).to(device)

    if resume is not None:
        print('resuming from: ' + resume)
        state = torch.load(resume)
        trial.load_state_dict(state)
        trial.replay()
    elif model_file is not None:
        trial.state[torchbearer.EPOCH] = 0
        trial.state[torchbearer.METRICS] = {}
        torchbearer.callbacks.MostRecent(model_file).on_checkpoint(trial.state)
        trial.state.pop(torchbearer.EPOCH)
        trial.state.pop(torchbearer.METRICS)

    history = None
    if trainloader is not None:
        history = trial.run(epochs, verbose=verbose)
        # also just save the plain final model weights
        torch.save(model.state_dict(), model_file)
    metrics = trial.evaluate(data_key=torchbearer.TEST_DATA)

    return trial, history, metrics


def evaluate_model(model, test_loader, metrics, extra_callbacks=None,
                   device='auto', verbose=0):
    device = get_device(device)

    cb = []
    if extra_callbacks is not None:
        if not isinstance(extra_callbacks, (list, tuple)):
            extra_callbacks = [extra_callbacks]
        cb.extend(extra_callbacks)

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    return (torchbearer.Trial(model, None, None, metrics=metrics, callbacks=cb,
                              verbose=verbose)
            .with_val_generator(test_loader).to(device).evaluate())


def load_model(model_file, model, device='auto'):
    """
    Load a model from a saved torchbearer trial file
    :param model_file: the saved file
    :param model: the model to populate
    :param device: the device
    :return: the model with loaded weights
    """
    device = get_device(device)
    model = model.to(device)
    weights = torch.load(model_file, map_location=device)[torchbearer.MODEL]
    model.load_state_dict(weights)

    return model


def _parse_schedule(sched):
    if '@' in sched:
        factor, schtype = sched.split('@')
        factor = float(factor)
    else:
        factor, schtype = None, sched

    step_on_batch = False
    if schtype.endswith('B') or schtype.endswith('b'):
        step_on_batch = True
        schtype = schtype[:-1]

    if schtype.startswith('every'):
        step = int(schtype[5:])
        return torchbearer.callbacks.StepLR(step, gamma=factor,
                                            step_on_batch=step_on_batch)
    if schtype.startswith('inv'):
        gamma, power = (float(i) for i in schtype[3:].split(","))
        return torchbearer.callbacks.LambdaLR(
            lambda i: (1 + gamma * i) ** (- power), step_on_batch=step_on_batch)
    elif schtype.startswith('['):
        milestones = json.loads(schtype)
        return torchbearer.callbacks.MultiStepLR(milestones, gamma=factor,
                                                 step_on_batch=step_on_batch)
    elif schtype == 'plateau':
        return torchbearer.callbacks.ReduceLROnPlateau(factor=factor,
                                                       step_on_batch=step_on_batch)

    assert False


def parse_learning_rate_arg(learning_rate: str):
    """
    Parse a learning rate argument into an initial rate and an optional
    scheduler callback.

    Examples:
        0.1                        --  fixed lr
        0.1*0.2@every10            --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@every10B           --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@[10,30,80]         --  decrease by factor=0.2 at epochs 10,
        30, and 80
        0.1*0.2@[10,30,80]B        --  decrease by factor=0.2 at epochs 10,
        30, and 80
        0.1*0.2@plateau            --  decrease by factor=0.2 on validation
        plateau
        0.1*inv0.0001,0.75B        --  decrease by using the old caffe inv
        rule each batch

    Args:
        learning_rate: lr string

    Returns:
        tuple of init_lr, callback
    """
    sp = str(learning_rate).split('*')
    initial = float(sp[0])

    if len(sp) == 1:
        return initial, None
    elif len(sp) == 2:
        return initial, _parse_schedule(sp[1])



