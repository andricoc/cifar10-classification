# Cifar10 classification
 Small school project to do classification using deep learning on cifar-10 dataset. This time, we will attach some additional layer to an existing model (MobileNetV2).
 The custom model show an accuracy of 82%.
 The code is in [classification_cifar10_mobilenetv2.ipynb ](https://github.com/andricoc/cifar10-classification/blob/main/classification_cifar10_mobilenetv2.ipynb)
 
 ## Overview
 CIFAR-10  is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

Data can be checked on https://www.kaggle.com/c/cifar-10 or the owner website on https://www.cs.toronto.edu/~kriz/cifar.html

The data are quite clean, not many preprocess are implemented. The data is normalized and resized.
```python
# Normalize pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
```

```python
# Upsize all training and testing images to 96x96 for use with mobile net
minSize = 96 #minimum size requried for mobileNet
resized_train = []
resized_test = []
for img in train_images:
    resized_train.append(cv2.resize(img, dsize=(minSize, minSize), interpolation=cv2.INTER_CUBIC))
for img in test_images:
    resized_test.append(cv2.resize(img, dsize=(minSize, minSize), interpolation=cv2.INTER_CUBIC))
train_images = np.asarray(resized_train,dtype='float32')
test_images = np.asarray(resized_test,dtype='float32')
```

MobileNetV2, a well-known existing model are used as the base. Then, custom layers will be added at the end of base model.

## Dataset Visual
![image](https://user-images.githubusercontent.com/63791918/229191205-2e7c1f03-924c-4f4b-b40b-dfa68bab08b9.png)

## Method
For the base model, MobileNetV2 are used. Not to forget, we should make the base model not trainable.
```python
base_model = applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False
```

After that custom layers are added at the end of the base model.
```python
model = Sequential([base_model,
                    layers.Dense(1000, activation='relu'),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(50, activation='relu'),
                    layers.Dense(10, activation='softmax')])
```

Add on some additional parameters before fitting the data. Compile the model with appropriate Loss function.
During fitting, batch size are set to 8 due to the large dataset. Can be adjusted accordingly
```python
history = model.fit(train_images, train_labels, epochs=10, batch_size = 8)
```

## Result
The model shows an accuracy of 82% and loss of 62% on the test data.
![image](https://user-images.githubusercontent.com/63791918/229191521-a94aefa4-a329-44e4-a7b7-e72366d99033.png)

### Classification Visualization
![image](https://user-images.githubusercontent.com/63791918/229195201-f9d1e0ae-841c-4c97-b8f4-0ced95b5194e.png)

### Accuracy and Loss progress
![image](https://user-images.githubusercontent.com/63791918/229195436-7345c906-7e5d-456d-9ddb-2a2fdc2f2458.png)

![image](https://user-images.githubusercontent.com/63791918/229195486-cf1cfdd3-297e-430b-b3e7-a43ddaab1ef8.png)

