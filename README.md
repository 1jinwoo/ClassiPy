# ClassiPy
## Features
- Takes search string as input and output category for the product that the search string is aimed to find.
- Best-performing model uses neural network with one layer with 330 neurons
- ClassiPy categorizes search strings via deep learning. Its general purpose is to train with a deep learning neural network to categorize key word descriptions. It can then return the correct category given key words at high accuracy. Real life applications of ClassiPy include search engines, online shopping, social media interpretation. We used ClassiPy to solve HCL's search string categorization challenge. By categorizing over 11,000 search strings into 63 categories, we trained deep learning neural network models, built front-end and back-end frameworks, predicted categories using real-time trained model, and returned results with extremely high accuracy and precision ( >87%). We also reduced standard deviation and variance in model. More details are included in the presentation.

## Dependencies
- NumPy 1.15.4
- Pandas 0.23.4
- TensorFlow 1.12.0
- Keras 2.2.4
- sklearn 0.20.1
- flask 1.0.2

## Contributors
- Justin Won (@1jinwoo)
- Jack Shi (@junyaoshi)
- Samuel Fu (@samuelfu)

## How to run
1. `cd` to wanted directory.
2. `git clone https://github.com/1jinwoo/ClassiPy` to clone.
3. Make sure the dependencies are installed. We recommend using Anaconda distribution of Python.
4. `cd` into the repository.
5. Run `python app.py`.
6. The web interface will launch in localhost. Use the URL given by the terminal message.
