import os
import numpy as np
import cv2
from traning import train
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

check_file = os.path.isfile('model.h5')
if not check_file:
    train()

model = load_model("model.h5")


def recognition(model, number):
    x = np.expand_dims(number, axis=0)
    res = model.predict(x)
    print(res)
    print(f'Распознанная цифра: {np.argmax(res)}')

    plt.imshow(number, cmap=plt.cm.binary)
    plt.show()


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


def paint(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=8)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=8)


def run():
    number = cv2.imread('number.jpg', cv2.IMREAD_GRAYSCALE)
    number = cv2.resize(number, (28, 28))
    number = np.array(number)
    number = number / 225

    recognition(model, number)


while 1:
    img = np.zeros((200, 200, 1), np.uint8)
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw', paint)
    while 1:
        cv2.imshow('test draw', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite('number.jpg', img)
            run()
            break
cv2.destroyAllWindows()