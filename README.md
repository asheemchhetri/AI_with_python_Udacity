# Artificial Intelligence with Python
## **Author**: Asheem Chhetri

![](https://img.shields.io/badge/Status-Complete-green.svg)

**Summary**: 
This nanodegree, talks about AI and its application, specifically images. This is a starting course, that provides a required foundation in the advanced field of **Deep Learning**, **Machine Learning**, **Computer Vision**, etc.

**Content**:

1. Quick summary of tools used
2. Project Discussion
3. Repository File Explaination
    - train.py
    - predict.py
    - checkpoint.pth
    - image Classifier Project.ipynb
    - image Classifier Project **Optimized**.ipynb
4. Terminal Usage
5. Data Discussion
    - Training
    - Testing
    - Validation
6. Result Discussion
    - Accuracy Comparison
    - **Prediction Results**
7. Practical Application
8. Where from here?
9. Completion Certificate
10. License

---

**1. Quick Summary of tools used**:

This project was completed using python 3 as coding enviornment, while major libraries were as follows:

- NumPy: Used to manipulate images, and to handle huge arrays thus making mathematical calculation less time consumable.
- PIL: Used to interact with images.
- torchvision(PyTorch): Used to load datasets, use *model architectures*, transformations, etc.
- time: Python module, to measure time took to train our model.
- json: Used to parse json file that contained information of *flower* names with relation to category

I also used *Jupyter Notebook*, to write my code, as it makes it very easy to focus only on code, rather than libraries, dependencies, etc.

**Important**: Model training is very resource intensive process, which can be completed in *CPU*, but it can take days to train a model that way. Other alternative is *GPU*, which speeds up the mathematical computation, and drastically reduces time required to train a model from days to few hours (depends on your model architecture). I had access to *56 hour* gpu resource available on Udacity server, while other alternatives are:

1. Google Colab: CLoud GPU service. I actually performed model optimization on this platform. User friendly, but has 12 hour limit on GPU usage, best way to use it through saving model in various checkpoints, and one can then use for various hours.
2. Kaggle: I submitted my model there as a part of fun competition. I did not try the GPU there, as I loaded my model, and perform prediction on CPU.

![PyTorch](/assets/pytorch.png)

![](https://img.shields.io/badge/version-0.3-blue.svg)

> [Image Source](https://www.analyticsindiamag.com/9-reasons-why-pytorch-will-become-your-favourite-deep-learning-tool/)

![Jupyter](/assets/jupyter.png)

![](https://img.shields.io/badge/version-5.7.4-blue.svg)

> [Image Source](https://hackernoon.com/10-tips-on-using-jupyter-notebook-abc0ba7028a4)

![Python](/assets/python.png)

![](https://img.shields.io/badge/version-3.6-blue.svg)

> [Image Source](https://commons.wikimedia.org/wiki/File:Python_logo_and_wordmark.svg)

---

**2. Project Discussion**:

It is expected, that in next few upcoming years software developers or hardware designers in field of IOT, computer architecture, etc would have incorporate **Deep Learning** models into day-to-day application, to perform activities such as:

1. Image Recognition
2. Object Detection
3. Hardware Failure Detection, etc.

In this *Nano Degree* program, I implemented an **Image Classification Application**, that trained a *Deep Learning* model on a set of **flower** images, and then used the trained model to classify the images. How well the classification is done, is based on predicition accuracy of our *trained model*.

---

**3. Repository File Explaination**:

- image Classifier Project.ipynb

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This **Jupyter Notebook** consists of model training code, that made use of *model architecture:* **densenet121**. It also specifies way to overwrite pre-existing model *classifier*, saving the checkpoint, loading the checkpoint and perform **Class Prediction** on an unseen image data. With this trained model, the model accuracy was around **90%**.

Classifier Code Snippet =>

```python
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout',  nn.Dropout(0.5)),
                          ('fc2', nn.Linear(500, 200)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(200, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
```

Criterion & Optimizer Code Snippet =>

```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate, amsgrad=True)
```

- image Classifier Project **Optimized**.ipynb

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This **Jupyter Notebook** consists of **Optimized** model training code, that made use of *model architecture:* **densenet161**(which has more *convolutional layers*). It also specifies way to overwrite pre-existing model *classifier*, saving the checkpoint, loading the checkpoint and perform **Class Prediction** on an unseen image data. With this trained model, the model accuracy was around **99%**. Also I made use of **step-LR** scheduler to adapt new *lear-rate* at every 5 epochs(iterations). This way we are able to train better. Last but not the least, I also **unfroze** the entire model from first checkpoint model, and retrained it much lower learn-rate, which further improved the accuracy!

Classifier Code Snippet =>

```python
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 600)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(600, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
```

Criterion, Optimizer & Scheduler Code Snippet =>

```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate, amsgrad=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

Unfreezing before *trained* model, before retraining it =>

```python
model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.require_grad = True
```
**Note**: It is never or nearly impossible to acheive 100% accuracy, as there always be some model that performs better on some constraints, while perform badly on others. It also depends on data we are training with, GPU availability, etc!

- train.py

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This python file holds the simialr code as in the jupyter notbeook, but now it has more practical usability by breaking the code in seperate module. This file **trains** the model, and save it as a corresponding checkpoint.pth file in cloud on local disk, so it can be used in other applications. Also for faster training, it is suggested to run this code, with **GPU** enabled.

**Note**: For higher functionalaity, it is important to convert the **Python** code into **C++** code, before deploying it. This new feature is supported in PyTorch 1.0, through the support of *JIT(Just In Time)* compiler. 

- predict.py

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This python file loads the checkpoint created by *train.py*, and passes an *image* as an input, which then responds with correct **class prediction** of that image. This prediction, depends on how *well* your model is trained!

- checkpoint.pth

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This file is what created by **train.py**, as it is bad idea to retrain a model again and again before using it! This way, once a model is trianed, it can be saved into external file. THis file can be used by any application, that knows how to load it, and it can be then succesfully used within various applications. Best part is, it can perform predictions on **CPU** resource, thus reducing the necessity on *consumer side* to have access to powerful **GPU**.

Checkpoint creation Code Snippet =>
```python
# Hyperparameters depends on your design, and model architecture used.
# This code snippet is based on densenet121
checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': arch,
              'classifier': classifier,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'learn_rate': learn_rate,
              'class_to_idx': train_data.class_to_idx,
             }

model_save_name = 'checkpoint_optimized.pt'
path = F"/content/gdrive/My Drive/Udacity/PyTorch_scholarship/final_project/{model_save_name}"
torch.save(checkpoint, path)
```

**4.** Terminal Usage

### Train

```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```

### Predict

```
python predict.py flowers/valid/27/image_06868.jpg
python predict.py flowers/test/58/image_02719.jpg --gpu
```

---

**5. Data Discussion**:

**Note**: These images are all **unique**, and is just a *subset* from entire dataset of *102* flower categories.

[Data Source](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

Flower Displayed below are from following class: **Pink Primrose**

- Training

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dataset is the one that is fed to our *model architecture*, and is **trained** based on looking at another data-set, which it does not know about. It is usually the **Validation** dataset. Based on how you write your code, this verification, helps us to understand accuracy of our currently being trained model.

![train_sample](/assets/train_1.jpg)

- Testing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dataset, that is usally not shown to our model, and is only fed to our model during prediction state. Based on the input, our model should be able to predict which flower is it!

![test_sample](/assets/test_1.jpg)

- Validation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dataset, is usually fed to our model, during training process. This way we can track the accuracy of our ongoing trained model.

![validation_sample](/assets/validation_1.jpg)

---

**6. Result Discussion**:

| File Name                                               | Model Architecture | Training Time(minutes) | Validation Accuracy(%)     | epochs | Scheduler         | Criterion | Optimizer | Learn Rate |
| :------------------------------------------------------ | :----------------: | :-------------------: | :------------------------: | :----: | :---------------: | :-------: | :-------: | :---------: |
| image Classifier Project.ipynb                          | densenet121        |  25                   | **89.09**                  | 6      | NO                | NLLLoss   | ADAM      | 0.001       |
| *Pass 1*: image Classifier Project **Optimized**.ipynb  | densenet161        | 196                   | **96.18**                  | 20     | YES (gamma = 0.1) | NLLLoss   | ADAM      | 0.001       |
| *Pass 2*: image Classifier Project **Optimized**.ipynb  | densenet161        | 148                   | **98.48**                  | 15     | YES (gamma = 10)  | NLLLoss   | ADAM      | 0.000001    |
| *Pass 3*: image Classifier Project **Optimized**.ipynb  | densenet161        | 49                    | **98.68**                  | 2      | YES (gamma = 1)   | NLLLoss   | ADAM      | 0.00005     |
| *Pass 4*: image Classifier Project **Optimized**.ipynb  | densenet161        | 9                     | **96.16**                  | 1      | YES (gamma = 1)   | NLLLoss   | ADAM      | 0.00001     |
| *Pass 5*: image Classifier Project **Optimized**.ipynb  | densenet161        | 19                    | **99.28**                  | 2      | YES (gamma = 1)   | NLLLoss   | ADAM      | 0.0000001   |

**Note**: As we can see, based on input, one may need to perform various passes to *acheive* higher accuracy, change models, learn_rate, etc.

Prediction Result for **densenet121** model, with **89.09%** accuracy:

![]()
![]()
![]()
![]()

---

Prediction Result for **densenet161** model, with **99.28%** accuracy:

![]()
![]()
![]()
![]()

---

Noticeable result **difference**:

| Model Used      | Accuracy (%) | Result |
| :-------------- | :----------: | :----:|
| densenet121     | 89.09        | ![](/assets/AI_nanodegree/pink_primrose.png) |
| densenet161     | 99.28        | ![](/assets/99_28/pink_primrose.png) |

---

**7. Practical Application**:

A possible application of this project, can be to train with **new** datasets(could be *health related*, *devices*, *cars*, etc). Upon prediction of new data, we can associate appropriate information related to an input.

Eg: Input is a *Car* model, based on prediction we can display *car* features to user, like engine specs, dealer locations, mileage, etc.

Possibilities are endless!

---

**8. Where from here?**

This is just beginning towards my AI journey, my plans are following:

**1. Machine Learning Engineering Nano Degree**: For in-depth understanding of *Supervised*, *Unsupervised* learning.<br/>
**2. Deep Learning Nano Degree**: For understanding CNN and RNN.<br/>
**3. Computer Vision Nano Degree**: For **real time** AI application.<br/>
**4. Deep Reinforcement Nano Degree**: One of most in-demand skill in AI field, to learn Deep Q-Learning.<br/>
**5. Artificial Engineering Nano Degree**: Understan *algorithms* applied to NLP, Computer vision, etc and be able to design new algorithm or optimze already available.<br/>
**6. Autonomous Flight Drones or Flying Cars(*yep no TYPO here*) Course**: What other best application can there be, than this!

Path is not **simple** and not **straight-forward**, but as long you are determined to do the hard work, there is nothing that can hold you back!

> "Just as machines made human muscles a thousand times stronger, machines will make the human brain a thousand times more powerful."

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SEBASTIAN THRUN, UDACITY

---

**9. Completion Certificate**

![Completion_Certificate](/assets/certificate.svg)

>[Certificate Link](https://confirm.udacity.com/RK94553P)

---

**10. License**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT) © Asheem Chhetri 2018-2019
