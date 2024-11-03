# FACEMASK_MTCNN
Face Mask Detection using MTCNN

## Prerequisites
![Gemini_Generated_Image_7h8lyq7h8lyq7h8l](https://github.com/user-attachments/assets/e215114c-a07f-48aa-9b25-b0c4e2626330)

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/Janeferdinant/FACEMASK_MTCNN/blob/main/requirements.txt)
 Cyber-Mask MTCNN facemask detector<code>blog.txt</code> [See here](https://www.blogger.com/blog/posts/8129491078230736872)




## Working
------------------

Run the python notebooks using jupyter notebook

Step -1 - Execute 1_data-preprocessing.ipynb 

Step -2 - Execute 2_training-face-mask-model.ipynb

Step -3 - Execute 3_detecting-mask-w-mtcnn.ipynb


FACEMASK_MTCNN
Face Mask Detection using MTCNN

Prerequisites
Software:

Python 3.x ([https://www.python.org/downloads/](https://www.python.org/downloads/))
Jupyter Notebook ([https://jupyter.org/](https://jupyter.org/))
Hardware:

A computer with a GPU (recommended for faster training)
Libraries: (These can be installed using the pip command)

You'll find a list of all required libraries in the requirements.txt file. Install them using:
<!-- end list -->

Bash
pip install -r requirements.txt


Cyber-Mask MTCNN facemask detector:

For further reading about MTCNN and its application in face mask detection, you can refer to this blog post: [blog ](https://www.blogger.com/blog/posts/8129491078230736872)(However, keep in mind that this link may not be the most relevant source. Feel free to search for more up-to-date information)
Working the Project
Instructions:

Clone the repository: (Assuming you have Git installed: [https://git-scm.com/](https://git-scm.com/)
Bash
git clone https://github.com/Janeferdinant/FACEMASK_MTCNN


Navigate to the project directory:
Bash
cd FACEMASK_MTCNN


Create a virtual environment (recommended for isolation):
Bash
python -m venv venv
source venv/bin/activate  
# Windows: venv\Scripts\activate


Install the project's dependencies:
Bash
pip install -r requirements.txt


Run Jupyter Notebook:
Bash
jupyter notebook


This will open a web interface where you can interact with the notebooks.
Data Preprocessing:

Run the first notebook: 1_data-preprocessing.ipynb This notebook preprocesses the dataset for training, including tasks like image resizing, data augmentation (optional), and potentially labeling images if not already labeled.
Training the Face Mask Detection Model:

Run the second notebook: 2_training-face-mask-model.ipynb This notebook trains the MTCNN model using the preprocessed data. Training time can vary depending on your hardware resources and dataset size.
Detecting Faces with Masks:

Run the third notebook: 3_detecting-mask-w-mtcnn.ipynb This notebook demonstrates how to use the trained model to detect faces and classify whether they are wearing masks or not on new images.
