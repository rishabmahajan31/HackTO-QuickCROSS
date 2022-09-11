import os

import sys

from regex import D 
sys.path.append('/usr/local/lib/python3.6/site-packages')
from flask import Flask,render_template,Response
import cv2
import time
import argparse


from detector import DetectorTF2

app=Flask(__name__)
cap=cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def DetectFromVideo():
    
	cap = cv2.VideoCapture(0)
	while (cap.isOpened()):
		ret, img = cap.read()
		if not ret: break

		timestamp1 = time.time()

		if args.webcam == True:
			img = rescale_frame(img, percent=150)

		det_boxes = detector.DetectFromImage(img)
		elapsed_time = round((time.time() - timestamp1) * 1000) #ms
		print("Detection time in ms: ", elapsed_time)
		img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)

		cv2.imshow('TF2 Detection', img)
		if cv2.waitKey(1) == 27: break


		ret,buffer=cv2.imencode('.jpg',img)
		frame=buffer.tobytes()
		yield(b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

	# print("Average execution time: ", (sum(results_list)/len(results_list)))
	# result_file.close()
	cap.release()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
parser.add_argument('--model_path', help='Path to frozen detection model',
					default='D:/College/TF_submission/Results/faster_rcnn_with_da/faster_rcnn_resnet_v1_georgian/exported_model/saved_model')
parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
					default='label_map.pbtxt')
parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
					type=str, default=None) # example input "1,3" to detect person and car
parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)

parser.add_argument('--webcam', help='Path to input video)', action='store_true', default='True')
args = parser.parse_args()

id_list = None
if args.class_ids is not None:
	id_list = [int(item) for item in args.class_ids.split(',')]

# if args.save_output:
# 	if not os.path.exists(args.output_directory):
# 		os.makedirs(args.output_directory)

# instance of the class DetectorTF2
detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)








@app.route('/video')
def video():
    return Response(DetectFromVideo(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)