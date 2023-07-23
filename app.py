from flask import Flask, render_template, request
import os
import torch
import pyiqa
import numpy as np
from skimage import io, color, util, measure
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def calculate_iqa_score(filename):
    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create metric with default settings
    iqa_metric = pyiqa.create_metric('niqe', device=device)

    # Note that gradient propagation is disabled by default. Set as_loss=True to enable it as a loss function.
    iqa_loss = pyiqa.create_metric('niqe', device=device, as_loss=True)

    score_clear = iqa_metric(filename)

    print("Clear image score for {} is {}".format(filename, score_clear))
    return score_clear


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part in the request"

        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty part without a filename
        if file.filename == '':
            return "No selected file"

        if file:
            # Save the uploaded file to the specified directory in the upload folder
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = io.imread(file_path)

            # Step 2: Convert the image to RGB if it's grayscale
            if image.ndim == 2:
                image = color.gray2rgb(image)

            # Step 3: Create a binary mask for pixels with RGB values
            threshold_value = 235
            binary_mask = np.all(image >= threshold_value, axis=-1)

            # Step 4: Convert the binary mask to a data type that can be used as a mask
            binary_mask = util.img_as_bool(binary_mask)

            # Step 5: Use the binary mask to isolate the area with the target RGB values
            isolated_area = np.zeros_like(image)
            isolated_area[binary_mask] = image[binary_mask]

            # Step 6: Perform connected component analysis to identify blobs in the isolated area
            labeled_area, num_labels = measure.label(binary_mask, return_num=True)
            blob_sizes = np.bincount(labeled_area.ravel())

            # Step 7: Define the threshold for minimum blob size (you can adjust this value)
            min_blob_size_threshold = 20

            # Step 8: Check if the isolated area blob is of a certain size
            if blob_sizes[num_labels - 1] >= min_blob_size_threshold:
                print("The isolated area contains glare.")
                glarepresent = "glare is present"
            else:
                print("The isolated area does not have glare.")
                glarepresent = "glare is not present"

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image)
            axes[0].set_title('Original Image')

            axes[1].imshow(isolated_area)
            axes[1].set_title('Isolated Areas')

            plt.show()

            score = calculate_iqa_score(file_path)

            return f"File '{filename}' successfully uploaded, clarity score is {score}, {glarepresent}"
    return "Something went wrong"


if __name__ == '__main__':
    app.run(debug=True)
