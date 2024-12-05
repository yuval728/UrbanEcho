# Urban Sound Classification

This project focuses on classifying urban sounds using deep learning techniques. The goal is to accurately identify different types of sounds commonly found in urban environments.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Urban sound classification is a challenging task due to the diverse nature of sounds in urban areas. This project aims to build a robust model that can classify various urban sounds such as car horns, sirens, and street music.

## Dataset
The dataset used for this project is the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Yuval728/urbanecho.git
cd urban-sound-classification
pip install -r requirements.txt
```

## Usage
To run the pipeline, run the following command:

```bash
python main.py
```


## Model
The model is built using a convolutional neural network (CNN) architecture. It takes Mel-spectrograms of the audio clips as input and outputs the predicted class.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
