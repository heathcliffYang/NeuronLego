import cv2

def get_image(img_path):
    image = cv2.imread(img_path)
    # crop upper part
    image = image[:int(image.shape/2),:]
    print(image.shape)
    image = cv2.cvtColor(cv2.resize(image, (128, 64)), cv2.COLOR_BGR2RGB)
    image = image / 255.
    return image
