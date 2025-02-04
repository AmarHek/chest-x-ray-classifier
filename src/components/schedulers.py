from torch.optim import lr_scheduler as lrs


def get_scheduler(scheduler: str, optimizer_fn, **kwargs):
    if scheduler == "step":
        return lrs.StepLR(optimizer=optimizer_fn,
                          step_size=kwargs["lr_decay_iters"],
                          gamma=0.1)
    elif scheduler == "plateau":
        return lrs.ReduceLROnPlateau(optimizer=optimizer_fn,
                                     mode=kwargs["mode"],
                                     patience=kwargs["plateau_patience"],
                                     factor=0.1,
                                     threshold=0.01)
    elif scheduler == "cosine":
        return lrs.CosineAnnealingLR(optimizer=optimizer_fn,
                                     T_max=kwargs["n_epochs"],
                                     eta_min=0)
    elif scheduler == "cyclic":
        return lrs.CyclicLR(optimizer=optimizer_fn,
                            base_lr=kwargs["cyclic_lr"][0],
                            max_lr=kwargs["cyclic_lr"][1])
    elif scheduler == "exponential_decay":
        return lrs.ExponentialLR(optimizer_fn, gamma=kwargs["exponential_gamma"])
    else:
        raise NotImplementedError(f"Scheduler {scheduler} not implemented")
