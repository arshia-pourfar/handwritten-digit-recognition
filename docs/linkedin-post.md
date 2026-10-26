# LinkedIn Post (Draft)

> Copy-paste ready. Replace `<repository-url>` after you push the repository to GitHub.

---

I just finished a small computer vision project: recognizing handwritten digits not only on
MNIST, but on real photos of digits written on paper by friends.

The interesting part wasn't the model — it was the gap between a benchmark dataset and the
real world. MNIST images are 28×28, centered, and clean. A phone photo is none of those. So
on top of a small CNN (three Conv2D blocks, early stopping, dropout), I built a classical
OpenCV preprocessing pipeline: Gaussian blur to reduce noise, automatic background inversion
based on the mean pixel value, binary thresholding, contour detection to crop the digit, an
aspect-ratio-preserving resize into a 20×20 box, and centering it on a 28×28 canvas — the
same convention used in the original MNIST preprocessing.

That pipeline is what makes the model usable outside the benchmark. The final version is
one clean app: `python app.py` opens a small menu where you can train either model (the
baseline MLP or the CNN) or drop digit photos into an `images/` folder and get a batch of
predictions — filename, predicted digit, and confidence — with a summary at the end.

The project is written in Python with TensorFlow/Keras and OpenCV, and both model
architectures are preserved from my original learning scripts so the progression from a
simple fully connected network to a CNN stays visible.

Code is on GitHub: <repository-url>

What I'd improve next: seeding runs for reproducibility, data augmentation, and a
confidence threshold that flags uncertain predictions.

#ComputerVision #DeepLearning #MachineLearning #Python #OpenCV #TensorFlow #AI
