import argparse
from pathlib import Path
import torch
from algorithm.yolov5.models.common import DetectMultiBackend
from algorithm.yolov5.utils.dataloaders import LoadImages
from algorithm.yolov5.utils.general import (Profile, check_img_size,
                           increment_path, non_max_suppression,  scale_boxes)
from algorithm.yolov5.utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        input_image,
        weights='yolov5s.pt',  # model path or triton URL
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project='',  # save results to project/name
        name='',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir  if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    stride, _, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, image=input_image,stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    for path, im, _, _, _ in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        det=pred[0]
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], input_image.shape).round()
        det=det.tolist()
        if len(det):
            for i in range(len(det)):
                det[i]=det[i][0:4]
                det[i]=[round(j) for j in det[i]]
        else:
            print("No detection")
        return det


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='algorithm/yolov5/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='algorithm/yolov5/detect.jpg', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', default=True,action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default='', help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True,action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def detect(image):
    result=run(input_image=image,**vars(parse_opt()))
    centers=[]
    for i in range(len(result)):
        tmp=result[i]
        center=[(tmp[0]+tmp[2])//2,(tmp[1]+tmp[3])//2]
        centers.append(center)
    return centers



