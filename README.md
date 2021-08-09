# Real-time Sign Language Detection

Real-time Sign Language Detection is a Python based project that provides all the steps required to produce the dataset required to train a nural network, train it and later use it for gesture recognition in real-time.
The input of gestures is only considered valid if a face is detected withing the image frame containing the gesture, else the input is ignored.

## Installation

The version of Python used is Python 3.9.6, which can be installed following this guide:
[Python install](https://www.python.org/downloads/)

The version of OpenCV used for this project is version 4.5.3, which can be installed using the shell command:
```bash
pip install opencv-python
```

For installing Jupyter Notebook, use the shell command:
```bash
pip install notebook
```
Be sure to activate the local server necessary to run jupyter notebooks, using the shell command:
```bash
jupyter notebook
```

For installing Tensorflow, required to train the neural network, I suggest to check out the official guide:
[Tensorflow installation](https://www.tensorflow.org/install)

For installing LabelImg and learning how to use it, I suggest the GitHub repository of the project:
[LabelImg GitHub](https://github.com/tzutalin/labelImg)

## Usage

When not specifically stated, all shell commands should be run from the root directory of the project.

To generate the dataset to be used to train a model, refer to the Jupyter notebook [GenerateDataset.ipynb](GenerateDataset.ipynb)

To label the dataset images and train the model, refer to the Jupyter notebook [LabelDataset.ipynb](LabelDatasetAndTrain.ipynb)

To run the sign detection algorithm, either refer to the Jupyter notebook [SignDetection.ipynb](SignDetection.ipynb), or run the python script using the shell command:
```bash
python SignDetection.py
```

## Contributing
Contribution to the project is welcome, although the project won't be maintained in the future by the development team.

## Authors and Acknowledgement

The project has been developed by Angela D'Antonio, Federica Moro and Marzio Vallero as part of the Computer Engineering Masters exam "Image Processing and Computer Vision", taught by professors B. Montrucchio and L. De Russis, during the academic year 2020/2021 at the Polythecnic of Turin.

## License
For right to use, copyright and warranty of this software, refer to the file [License.md](License.md)