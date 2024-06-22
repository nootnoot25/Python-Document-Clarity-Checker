# Flask Image Quality Assessment and Glare Detection

This is a Flask web application that allows users to upload images, check for glare, and calculate an Image Quality Assessment (IQA) score using the Naturalness Image Quality Evaluator (NIQE).

## Features

- Upload image files.
- Convert grayscale images to RGB.
- Isolate areas with potential glare based on RGB values.
- Perform connected component analysis to detect glare.
- Calculate an IQA score using the NIQE metric.
- Display the original and isolated areas in images.

## Requirements

- Python 3.7+
- Flask
- PyTorch
- pyiqa
- NumPy
- scikit-image
- Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure the `uploads` directory exists in the root of your project:

    ```bash
    mkdir uploads
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

3. Upload an image file using the provided form. The application will:
    - Save the uploaded file.
    - Convert the image to RGB if it's grayscale.
    - Create a binary mask for pixels with high RGB values (potential glare).
    - Isolate the area with glare.
    - Perform connected component analysis to identify blobs.
    - Display the original image and the isolated areas.
    - Calculate the IQA score and check for glare presence.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
