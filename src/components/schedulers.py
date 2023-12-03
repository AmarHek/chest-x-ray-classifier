from torch.optim import lr_scheduler as lrs


def get_scheduler(scheduler: str, optimizer, mode: str = "min", **kwargs):

    if scheduler == "none":
        return None
    # TODO: implement linear and step
    elif scheduler == "linear":
        return lrs.StepLR(**kwargs)
    elif scheduler == "step":
        return lrs.StepLR(**kwargs)
    elif scheduler == "plateau":
        return lrs.ReduceLROnPlateau(optimizer=optimizer,
                                     mode=mode,
                                     patience=kwargs["plateau_patience"],)
    elif scheduler == "cosine":
        # TODO
        return lrs.CosineAnnealingLR(**kwargs)
    elif scheduler == "cyclic":
        return lrs.CyclicLR(optimizer=optimizer,
                            base_lr=kwargs["cyclic_lr"][0],
                            max_lr=kwargs["cyclic_lr"][1])
    elif scheduler == "exponential_decay":
        return lrs.ExponentialLR(optimizer, gamma=kwargs["exponential_gamma"])
    else:
        raise NotImplementedError(f"Scheduler {scheduler} not implemented")
