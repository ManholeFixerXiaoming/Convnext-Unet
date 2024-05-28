# ConvNext-UNet-X2&X4 for Short-Term Rainfall Radar Echo Extrapolation（Nowcasting）

This project aims to use the ConvNext-UNet model for radar echo nowcasting, specifically for short-term heavy rainfall prediction. The project provides X2 and X4 scale models.

## Features

- **High Accuracy**: Combines the advantages of ConvNext and UNet models to improve the accuracy of radar echo extrapolation.
- **Multi-Scale**: Provides X2 and X4 scale models to suit different application scenarios.
- **Ease of Use**: Simple interface design for easy and quick adoption.

## Prerequisites

Please see the [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt) for the required libraries and dependencies.

## Usage

trainBN_convnext_X2.py is the main file, where you can load both the X2 and X4 models. The traindata.7z file contains a portion of the training data for testing the models.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

The ConvNext layers used in this project are sourced from [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) by Facebook Research.


