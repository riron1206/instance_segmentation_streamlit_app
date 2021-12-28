import os, shutil
import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


@st.cache(persist=True)
def initialization(st_score_th=0.5):
    """Loads configuration and model for the prediction.

    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.

    """
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = st_score_th
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


@st.cache
def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img


@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.

    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them

    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes",
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.

    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.

    """
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Get the indexes
    idx = np.nonzero(mask)

    # Get the current Instance values
    pred_boxes = outputs['instances'].pred_boxes
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks
    scores = outputs['instances'].scores

    # Get them as a dictionary and leave only the desired ones with the indexes
    out_fields = outputs['instances'].get_fields()
    out_fields['pred_boxes'] = pred_boxes[idx]
    out_fields['pred_classes'] = pred_classes[idx]
    out_fields['pred_masks'] = pred_masks[idx]
    out_fields['scores'] = scores[idx]

    return outputs


def main():

    # サイドバー スライダー
    st_score_th = st.sidebar.slider("Threshold", 0.0, 1.0, step=0.01, value=0.5)

    # Initialization
    cfg, predictor = initialization(st_score_th)

    # Streamlit initialization
    st.title("Instance Segmentation (mask_rcnn_R_50_FPN_3x)")

    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:

        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)

        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Processed Image', use_column_width=True)

        # 推論結果の画像を保存する
        out_img_path = "output/inference_" + str(uploaded_img.name)
        cv2.imwrite(out_img_path, out_image)
        # ダウンロード
        # https://zenn.dev/ohtaman/articles/streamlit_tips
        st.download_button(
            'ダウンロード', open(out_img_path, 'br'), out_img_path
        )
    else:
        img_url = "inference_0a4f658ae77b7e4209e22b79fe1c28cb.jpg"
        st.image(
            img_url,
            caption="Output sample. Please upload image.",
            use_column_width=True,
        )


if __name__ == '__main__':
    main()