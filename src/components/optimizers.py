import torch.optim as optim
import libauc.optimizers as aucoptim


# TODO add more optimizer options (in params)
def get_optimizer(optimizer, learning_rate, model, **kwargs):
    optimizer = optimizer.lower()

    if optimizer == "adadelta":
        return optim.Adadelta(model.parameters(),
                              lr=learning_rate)
    elif optimizer == "adagrad":
        return optim.Adagrad(model.parameters(),
                             lr=learning_rate)
    elif optimizer == "adam":
        return optim.Adam(model.parameters(),
                          lr=learning_rate)
    elif optimizer == "adamw":
        return optim.AdamW(model.parameters(),
                           lr=learning_rate)
    elif optimizer == "adamax":
        return optim.Adamax(model.parameters(),
                            lr=learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(),
                         lr=learning_rate)
    elif optimizer == "asgd":
        return optim.ASGD(model.parameters(),
                          lr=learning_rate)
    elif optimizer == "nadam":
        return optim.NAdam(model.parameters(),
                           lr=learning_rate)
    elif optimizer == "radam":
        return optim.RAdam(model.parameters(),
                           lr=learning_rate)
    elif optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(),
                             lr=learning_rate)
    elif optimizer == "rprop":
        return optim.Rprop(model.parameters(),
                           lr=learning_rate)
    elif optimizer == "lbfgs":
        return optim.LBFGS(model.parameters(),
                           lr=learning_rate)
    elif optimizer == "pesg":
        return aucoptim.PESG(model.parameters(),
                             loss_fn=kwargs['loss'],
                             lr=learning_rate,
                             momentum=0.9,
                             epoch_decay=0.003,
                             weight_decay=0.0001)
    elif optimizer == "pdsca":
        return aucoptim.PDSCA(model.parameters(),
                              loss_fn=kwargs['loss'],
                              lr=learning_rate,
                              beta1=0.9,
                              beta2=0.9,
                              margin=1.0,
                              epoch_decay=0.002,
                              weight_decay=0.0001)
    elif optimizer == "soap":
        return aucoptim.SOAP(model.parameters(),
                             lr=learning_rate)
    elif optimizer == "sopa":
        return aucoptim.SOPA(model.parameters(),
                             loss_fn=kwargs['loss'],
                             lr=learning_rate,
                             weight_decay=0.0001)
    elif optimizer == "sopas":
        return aucoptim.SOPAs(model.parameters(),
                              loss_fn=kwargs['loss'],
                              lr=learning_rate,
                              weight_decay=0.0001)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented!")
