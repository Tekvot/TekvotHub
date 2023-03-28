import sys
import warnings
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'detectors')
sys.path.insert(2, 'detectors/yolov7')

# from detectors.maskrcnn import MaskRCNN
from detectors.yolo7 import Yolo7
import cv2


class Detector:
    def __init__(self, detector_name, weights, data, device):

        print("Detector in building............!!!")

        self.model = None

        # '''''''''' Warning: NoneType variable '''''''''''
        if detector_name is None:
            warnings.warn('detector_name is a NoneType object')
            return

        # # ''''''''''''''''' Mask RCNN ''''''''''''''''
        # elif detector_name == 'maskrcnn':
        #     self.model = MaskRCNN(weights= './detectors/maskrcnn/model_final.pth',
        #                           data="./detectors/maskrcnn/IS_cfg.pickle", 
        #                           device='cuda:0')

        # '''''''''''''''''' YoloV7 '''''''''''''''''''
        elif detector_name == 'yolo7':
            self.model = Yolo7(weights = weights, 
                               data = data, 
                               device = device)

        # '''''''''' Warning: Not available model '''''''''''
        else:
            warnings.warn('Model is not in available')
            return


    def predict(self, input):
        if self.model is None:
            warnings.warn('self.Model is a NoneType object, please select an available model')
            return
        
        predictions = self.model.predict(input)
        return predictions


if __name__=='__main__':
    img = cv2.imread('./detectors/gallery/seedlings0.jpg')
    detector = Detector('yolo7', weights='./weights/yolov7-hseed.pt', data='./weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    predictions = detector.predict(img)

    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        
        
    cv2.imshow('awd',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

