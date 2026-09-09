import os, cv2, sys, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection2.src.utils.response import build_response
from sdks.novavision.src.base.model import Detection, BoundingBox
from capsules.ForegroundDetection2.src.models.PackageModel import PackageModel

class ForegroundDetection2(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.threshold = self.request.get_param("threshold")
        self.min_contour_area = self.request.get_param("minContourArea")
        self.detections = []

    def clean_mask(self, raw_mask):
        _, mask = cv2.threshold(raw_mask, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value.astype(np.uint8)
        
        # Gelen frame zaten maske ama 3 kanalli (BGR) olabilir. Griye cevir.
        if len(frame.shape) == 3:
            raw_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raw_mask = frame

        fg_mask = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.detections = []
        vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detection = Detection(
                boundingBox=BoundingBox(left=x, top=y, width=w, height=h),
                confidence=1.0,
                classId=0,
                classLabel="foreground",
                imgUID=self.uID,
                keyPoints=[]
            )
            self.detections.append(detection)

        img.value = vis
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
