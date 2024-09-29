# Chessboard Recogniser

This repository contains a Python project, implemented as a series of Jupyter notebooks, that uses computer vision techniques and a Convolutional Neural Network (CNN) to detect the FEN (Forsyth-Edwards Notation) string from an image of a chessboard.

## Demo
![alt text](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)

Try out a live demo of the chessboard recognizer on Hugging Face Spaces:
https://huggingface.co/spaces/salominavina/chessboard-recognizer

## Description

The project aims to accurately identify the pieces on a chessboard from a given image and represent the board state in the standard FEN format. This can be used for various applications, such as:

* **Chess game analysis:** Automatically extract the board position from images or videos of chess games.
* **Chess AI training:** Generate datasets of board positions with corresponding FEN strings for training chess AI models.
* **Chess notation tools:** Develop tools that can automatically convert images of chessboards into digital formats.

## Notebooks

The project is divided into four main Jupyter notebooks:

1. **`Chess_Piece_Classification_CNN.ipynb`:** This notebook contains the code for training a CNN model to classify individual chess pieces.
2. **`generating_tiles_computervision.ipynb`:** This notebook demonstrates how to use computer vision techniques to extract individual squares (tiles) from a chessboard image.
3. **`generating_training_data.ipynb`:** This notebook shows how to generate training data for the CNN model by extracting chess pieces from images and labeling them.
4. **`Predict_Chessboard_CNN.ipynb`:** This notebook uses the trained CNN model to predict the pieces on a chessboard image and generate the corresponding FEN string.

## How it Works

1. **Image preprocessing:** The input image is processed to enhance the chessboard features and remove noise. This may involve techniques like cropping, resizing, and color correction.
2. **Square detection:** The image is analyzed to identify the individual squares on the chessboard using edge detection and perspective transformation.
3. **Piece recognition:** Each square is classified using the pre-trained CNN model from `Chess_Piece_Classification_CNN.ipynb` to determine if it contains a piece and, if so, which piece it is.
4. **FEN generation:** The detected pieces and their positions are used to construct the FEN string, which represents the current state of the chessboard.

## Features

* **Accurate FEN detection:** The project aims for high accuracy in recognizing the pieces and generating the correct FEN string.
* **Robust to image variations:** The system should be able to handle images with different lighting conditions, angles, and resolutions.

## Requirements

The project requires the following libraries:

* Python 3.x
* OpenCV
* PyTorch
* Gradio
* PyQt5
* PyQtWebEngine
* Pillow
* NumPy
* Matplotlib

You can install these libraries using pip:

```bash
pip install opencv-python torch gradio PyQt5 PyQtWebEngine Pillow numpy matplotlib
```

## Usage

To run the project, follow these steps:

Launch Jupyter Notebook: jupyter notebook

Open the notebooks in the following order:

generating_training_data.ipynb (optional, only if you need to generate new training data)

generating_tiles_computervision.ipynb (to understand how tiles are extracted)

Chess_Piece_Classification_CNN.ipynb (to train the CNN model)

Predict_Chessboard_CNN.ipynb (to predict FEN from an image)

Run the cells in each notebook sequentially.

The notebooks will guide you through the process of training the model, loading an image, running the FEN detection algorithm, and displaying the results.

## Example

Input image: chessboard.jpg

Output FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

## Contributing

Contributions are welcome! If you have any ideas for improvements or find any bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
