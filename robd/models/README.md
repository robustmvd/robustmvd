# Models

robd contains implementations of depth estimation models. The usage of the models is described in the following.

## Usage

### Initialization

To initialize a model, use the `create_model` function:
```python
from robd import create_model
model = create_model(model_name, pretrained=True, weights=None)  # optional: model-specific parameters
```

#### Model variants
Some models support multiple variants/configurations. They are represented by different model names. 
For all list of all models including their configurations, use the `list_models` function:
```python
from robd import list_models
print(list_models())
```

#### Weights

If `pretrained` is set to True, the default pretrained weights for the model will be used. The default weights
are automatically downloaded at first use. 
Alternatively, custom weights can be loaded by providing the path to the weights with the `weights` parameter.

### Interface
The interface to use the models is:
```
pred, aux = model(images, poses, intrinsics, keyview_idx)  # optional: depth_range ; alternatively: model(**sample)
```
The input data are torch tensors with a prepended batch dimension, as described in the [data readme](../data/README.md).

The `pred` output is a dictionary which contains:
- `depth`: predicted depth map for the reference view
- `depth_uncertainty`: predicted uncertainty for the predicted depth map (optional)
The outputs are also torch tensors with a prepended batch dimension, e.g. `depth` has shape N1HW.

The `aux` output is a dictionary which contains additional model-specific outputs. These are only used for training 
or debugging and not further described here.

#### Resolution
Most models cannot handle input images at arbitrary resolutions. Models therefore internally downsize the images to
the next resolution that can be handled. 

The model output is often at a lower resolution as the input data.

## Wrapped Models
For completeness, we provide a few wrappers around existing model implementations. 
The usage of these models is a bit more involved, as it is required to download the original code 
and set up an appropriate environment. TODO.

### Usage
TODO
