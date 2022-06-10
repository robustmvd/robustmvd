# Robust Depth

**Rob**ust **D**epth (**robd**) is a framework and benchmark for robust depth estimation from multiple input views.
It contains implementations and weights of recent models, training/evaluation/inference scripts and benchmark results.

It supports multiple common settings:
- Multi-view depth (MVD): reconstruct depth map for a keyview from multiple unstructured, calibrated source views
- Multi-view stereo (MVS): reconstruct scene from multiple unstructured, calibrated views
- Depth-from-video (V2D): estimate depth maps for frames of a video sequence from a camera with known intrinsics

robd is mainly focused on models that are implemented in pytorch, but provides means for evaluating tensorflow models. 
## Setup

The code was tested with python 3.8 and PyTorch 1.9. To set it up, one can either use the `requirements.txt`
or the `setup.py`.

To install the requirements, run:
```bash
pip install -r requirements.txt
```

To install the package using the `setup.py`, run:
```bash
python setup.py install
```
The package can then be imported via `import robd`.

To use the dataloaders from robd, datasets need to be downloaded and some need to be preprocessed before 
they can be used. For details, see [robd/data/README.md](robd/data/README.md).

## Structure

The framework contains dataloaders, models, training/evaluation/inference scripts for multi-view depth estimation.
Further, it contains a set of tools for visualizing and reporting results.

The setup and interface of the dataloaders is explained in [robd/data/README.md](robd/data/README.md).

The setup and interface of the models is explained in [robd/models/README.md](robd/models/README.md).

The usage of the training, evaluation and inference scripts is explained below.

The usage of the tools is explained in [robd/tools/README.md](robd/tools/README.md).

## Evaluation script
Evaluation is done with the script `eval.py`, for example:
```bash
python eval.py --model robust_mvd --dataset eth3d --setting mvd --input poses --input intrinsics --output /path/to/output
```

The parameters `model`, `dataset` and `setting` are required. Note that not all models and datasets support all
evaluation settings. For an overview, see the [models](robd/models/README.md) and [data](robd/data/README.md) readme.

For further parameters, execute `python eval.py --help`.

## Programmatic evaluation

It is also possible to run the evaluation from python code, for example with:
```python
import robd
eval = robd.create_evaluation(evaluation_type="mvd", out_dir="/path/to/output", inputs=["intrinsics", "poses"])
dataset = robd.create_dataset("eth3d", "mvd", to_torch=True, input_size=(384, 576))
results = eval(dataset=dataset, model=your_model)
```

For further details, see the documentation. 

## Inference script
TODO

## Programmatic inference
`robd` models can be used programmatically, e.g.:
```python
import robd
model = robd.create_model("robust_mvd")
dataset = robd.create_dataset("eth3d", "mvd", to_torch=True, input_size=(384, 576))
sample = dataset[0]
pred = model(**sample)  # TODO: add visualization call
```

## Training script
TODO

## Compatability with tensorflow models
TODO

## Results

Results are provided for all models that are included in this framework. The results are obtained with the script
`eval.sh` and can be found in the `results` directory. 

TODO.

## Conventions
Within this package, we use the following conventions:
- all data is in float32 format
- all data on an image grid uses CHW format (e.g. images are 3HW, depth maps 1HW); batches use NCHW format 
(e.g. images are N3HW, depth maps N1HW, poses N144, etc.)
- models output predictions potentially at a downscaled resolution
- resolutions are indicated as (height, width) everywhere
- if the depth range of a scene is unknown, we consider a default depth range of (0.1m, 100m)
