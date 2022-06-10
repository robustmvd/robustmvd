from .registry import get_model


def create_model(name, pretrained=True, weights=None, **kwargs):
    """Create a model.

    Args:
        name (str): The name of the model to create.
        pretrained (bool): Whether to load the default pretrained weights for the model.
        weights (str): Path to custom weights to be loaded. Overrides `pretrained`.

    Keyword Args:
        **kwargs: Additional arguments to pass to the model.
    """
    model_entrypoint = get_model(name=name)
    model = model_entrypoint(pretrained=pretrained, weights=weights, **kwargs)
    return model
