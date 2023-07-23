from flask import Flask, render_template, request
import os
import torch
import pyiqa

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

            score = calculate_iqa_score(file_path)
            return f"File '{filename}' successfully uploaded, clarity score is {score} "
    return "Something went wrong"


if __name__ == '__main__':
    app.run(debug=True)
