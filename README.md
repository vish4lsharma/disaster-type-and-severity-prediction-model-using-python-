# Disaster Analyzer

## Overview
The **Disaster Analyzer** is a Python-based tool designed to detect and analyze natural disasters in images. It supports the detection of various disaster types, including:

- Flood
- Fire
- Hurricane
- Landslide
- Snowstorm

The tool utilizes computer vision techniques with OpenCV, PIL, and PyTorch for image processing and classification. It also incorporates parallel processing to speed up the analysis.

## Features
- **Automatic Disaster Detection**: Identifies and classifies disaster types present in an image.
- **Severity Assessment**: Determines the severity level of detected disasters based on predefined thresholds.
- **Mask Overlay Visualization**: Generates images with color-coded overlays for detected disaster regions.
- **Parallel Processing**: Enhances performance by analyzing multiple disaster types simultaneously.
- **Robust Error Handling**: Includes logging and exception handling for improved reliability.

## Installation
To use the Disaster Analyzer, ensure you have the required dependencies installed. You can install them using:

```sh
pip install torch torchvision pillow requests numpy opencv-python matplotlib
```

## Usage
The `DisasterAnalyzer` class provides an interface to analyze images from URLs. Below is a basic example:

```python
from disaster_analyzer import DisasterAnalyzer

analyzer = DisasterAnalyzer()
image_url = "https://example.com/disaster-image.jpg"
results = analyzer.analyze_image(image_url)
```

You can also specify the disaster types you want to analyze:

```python
results = analyzer.analyze_image(image_url, ['flood', 'fire'])
```

## Visualization
The tool provides visualizations of the detected disaster areas, overlaying masks onto the original image to highlight affected regions.

## Example
A sample execution of the script:

```sh
python disaster_analyzer.py
```

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, feel free to reach out to the project maintainers.

