from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


if __name__ == '__main__':
    pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
    modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
    IMG_PATH = '/home/sondn/DIY/StackGAN-Pytorch/output/images'

    weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model,
                           cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    # detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(IMG_PATH)

    faces = []
    for img in image_generator:
        image = np.array(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img_size, img_size)))
        faces.append(image)

    # predict ages and genders of the detected faces
    inp = np.array(faces)
    results = model.predict(inp)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    # draw results
    for i, d in enumerate(faces):
        label = "{}, {}".format(int(predicted_ages[i]),
                                "M" if predicted_genders[i][0] < 0.5 else "F")
        print(label)
        # draw_label(img, (d.left(), d.top()), label)

    # cv2.imshow(img)
    # key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

    # if key == 27:  # ESC
    #     break