import cv2

if __name__ == '__main__':
    img_stack = cv2.imreadmulti("data/tiffs/train-volume.tif")[1]
    label_stack = cv2.imreadmulti("data/tiffs/train-labels.tif")[1]

    for i in range(len(img_stack)):
        cv2.imwrite("data/train/images/" + str(i) + ".jpg", img_stack[i])

    for i in range(len(label_stack)):
        cv2.imwrite("data/train/masks/" + str(i) + ".jpg", label_stack[i])

    img_stack = cv2.imreadmulti("data/tiffs/test-volume.tif")[1]
    label_stack = cv2.imreadmulti("data/tiffs/test-labels.tif")[1]

    for i in range(len(img_stack)):
        cv2.imwrite("data/test/images/" + str(i) + ".jpg", img_stack[i])

    for i in range(len(label_stack)):
        cv2.imwrite("data/test/masks/" + str(i) + ".jpg", label_stack[i])
