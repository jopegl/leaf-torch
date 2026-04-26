import torch

def load_crossval_models(model_fn, device, model_paths):
    models = {}

    for fold, path in model_paths.items():

        print(f"Carregando fold {fold}: {path}")

        model = model_fn().to(device)

        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        models[fold] = model

    return models