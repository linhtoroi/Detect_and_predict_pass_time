import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
from typing import DefaultDict
import imutils
import time
from scipy import spatial
import cv2
from sklearn.model_selection import train_test_split
from src.model import Model

class LFramework(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LFramework, self).__init__()
        self.input_size = input_size
        self.model = Model(2, hidden_size, output_size)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.epoch = 50
        self.labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")
    
        self.weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
        self.configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

        self.videoPath = "data/video/"
        self.inputVideoPath = None
        self.outputVideoPath = None

        self.processedDataPath = 'data/processed/'
        
        self.preDefinedConfidence = 0.5
        self.preDefinedThreshold = 0.3

        self.list_of_vehicles = ["bicycle","car","motorbike","bus","truck", "train"]

        self.FRAMES_BEFORE_CURRENT = 10  
        self.inputWidth, self.inputHeight = 416, 416

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
            dtype="uint8")

    def defineVideoPath(self, data_name, file_extension):
        self.inputVideoPath = self.videoPath + 'input/' + data_name + file_extension
        self.outputVideoPath = self.videoPath + 'output/' + data_name + '.avi'

    def train(self, data_name, time_data):
        X, y = self.split_batch(data_name, time_data)
        X_train, X_val, y_train, y_val = train_test_split(X.unsqueeze(dim=1), y.unsqueeze(dim=1), test_size = 0.2, shuffle=False)
        loss_min = np.inf

        self.model.train()
        for i in range(self.epoch):
            for cnt in tqdm(range(len(X_train))):
                self.optimizer.zero_grad()
                output = self.model(X_train[cnt])
                loss = self.loss_fn(output, y_train)
                loss.backward()
                self.optimizer.step()


            self.model.eval()
            for cnt in range(len(X_val)):
                output = self.model(X_val[cnt])
                loss_2 = self.loss_fn(output, y_val)
            
            print('loss: train = {} valid = {}'.format(loss.item(), loss_2.item()))
            if loss_min > loss_2.item():
                loss_min = loss_2.item()
                state = {
                  'epoch': i,
                  'state_dict': self.model.state_dict(),
                  'optimizer': self.optimizer.state_dict(),
                  'loss': loss,
                }
                torch.save(state, 'model/model_best.tar')
                print('Save model to model/model_best.tar')

            self.model.train()

    def load_model(self):
        checkpoint = torch.load('model/model_best.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def predict(self, input):
        return self.model(input)

    def run(self, time_data):
        net_detect = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        
        ln = net_detect.getLayerNames()
        ln = [ln[i[0] - 1] for i in net_detect.getUnconnectedOutLayers()]

        videoStream = cv2.VideoCapture(self.inputVideoPath)
        video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1_line = 0
        y1_line = video_height//2
        x2_line = video_width
        y2_line = video_height//2

        previous_frame_detections = [{(0,0):0} for i in range(self.FRAMES_BEFORE_CURRENT)]

        num_frames, vehicle_count = 0, 0
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        writer = self.initializeVideoWriter(video_width, video_height, videoStream)
        start_time = int(time.time())

        #########
        total_frame = 0

        while True:
            total_frame += 1
            num_frames += 1

            boxes, confidences, classIDs = [], [], [] 
            vehicle_crossed_line_flag = False 

            start_time, num_frames = self.displayFPS(start_time, num_frames)
            (grabbed, frame) = videoStream.read()

            if not grabbed:
                break

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inputWidth, self.inputHeight),
                swapRB=True, crop=False)
            net_detect.setInput(blob)
            start = time.time()
            layerOutputs = net_detect.forward(ln)
            end = time.time()

            for output in layerOutputs:
                for i, detection in enumerate(output):
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.preDefinedConfidence:
                        box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                                    
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.preDefinedConfidence, self.preDefinedThreshold)

            self.drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

            vehicle_count, current_detections, vehicle_count_in_frame = self.count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)
            
            self.displayVehicleCount(frame, vehicle_count)

            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break	
            
            previous_frame_detections.pop(0)
            previous_frame_detections.append(current_detections)

            input_2 = np.full(len([vehicle_count_in_frame]), time_data)
            input = np.hstack((np.array([vehicle_count_in_frame]).reshape(len([vehicle_count_in_frame]),1),np.array(input_2).reshape(len(input_2),1))).astype('float32')
            data = torch.tensor(np.array(input).astype('float32')).unsqueeze(dim=0)
            self.model.eval()
            output = self.predict(data)
            for cnt, i in enumerate(output[0][0]):
                print("Sau {} giây nữa, cần bao nhiêu thời gian mới đi được hết quãng đường: {}".format(cnt*10+10, i))
            print("__________________________________________________________________________________________")


    def split_batch(self, data_name, time_data):
        f = open('{}{}.txt'.format(self.processedDataPath, data_name), 'r')
        input_1 = f.read().split('\n')
        input_1 = [int(i) for i in input_1[:-1]]
        input_2 = np.full(len(input_1), time_data)

        input = np.hstack((np.array(input_1).reshape(len(input_1),1),np.array(input_2).reshape(len(input_2),1))).astype('float32')

        f = open('{}label_{}.txt'.format(self.processedDataPath, data_name), 'r')
        target = f.read().split('\n')
        label = []

        for i in target[:-1]:
            for _ in range(290):
                m = i[1:-1].split(', ')
                label.append([float(j) for j in m])

        x = torch.tensor(np.array(np.array_split(input[:int(len(input)/290)*290],int(len(input)/290))).astype('float32'))
        y = torch.tensor(np.array(np.array_split(np.array(label[:int(len(input)/290)*290]),int(len(input)/290))).astype('float32'))

        return x, y

    def process_data(self, data_name, file_extension):
        self.defineVideoPath(data_name, file_extension)
        net_detect = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        
        ln = net_detect.getLayerNames()
        ln = [ln[i[0] - 1] for i in net_detect.getUnconnectedOutLayers()]

        videoStream = cv2.VideoCapture(self.inputVideoPath)
        video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1_line = 0
        y1_line = video_height//2
        x2_line = video_width
        y2_line = video_height//2

        previous_frame_detections = [{(0,0):0} for i in range(self.FRAMES_BEFORE_CURRENT)]

        num_frames, vehicle_count = 0, 0
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        writer = self.initializeVideoWriter(video_width, video_height, videoStream)
        start_time = int(time.time())

        #########
        total_frame = 0
        vehicle_count_in_frame = 0
        obj_pass = {}

        while True:
            total_frame += 1
            num_frames += 1

            boxes, confidences, classIDs = [], [], [] 
            vehicle_crossed_line_flag = False 

            start_time, num_frames = self.displayFPS(start_time, num_frames)
            (grabbed, frame) = videoStream.read()

            if not grabbed:
                break

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inputWidth, self.inputHeight),
                swapRB=True, crop=False)
            net_detect.setInput(blob)
            start = time.time()
            layerOutputs = net_detect.forward(ln)
            end = time.time()

            for output in layerOutputs:
                for i, detection in enumerate(output):
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.preDefinedConfidence:
                        box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                                    
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.preDefinedConfidence, self.preDefinedThreshold)

            self.drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

            vehicle_count, current_detections, vehicle_count_in_frame = self.count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)
            
            self.displayVehicleCount(frame, vehicle_count)

            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break	
            
            previous_frame_detections.pop(0)
            previous_frame_detections.append(current_detections)

            with open('{}{}.txt'.format(self.processedDataPath, data_name), 'a') as f:
                f.write(str(vehicle_count_in_frame)+'\n')

            obj_pass = self.extract_startframe_and_endframe(previous_frame_detections, obj_pass, total_frame, data_name)

        label = self.process_label(data_name, sourceVideofps)

    def process_input(self, data_name, file_extension):
        self.defineVideoPath(data_name, file_extension)
        net_detect = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        
        ln = net_detect.getLayerNames()
        ln = [ln[i[0] - 1] for i in net_detect.getUnconnectedOutLayers()]

        videoStream = cv2.VideoCapture(self.inputVideoPath)
        video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1_line = 0
        y1_line = video_height//2
        x2_line = video_width
        y2_line = video_height//2

        previous_frame_detections = [{(0,0):0} for i in range(self.FRAMES_BEFORE_CURRENT)]

        num_frames, vehicle_count = 0, 0
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        writer = self.initializeVideoWriter(video_width, video_height, videoStream)
        start_time = int(time.time())

        #########
        total_frame = 0
        vehicle_count_in_frame = 0

        while True:
            total_frame += 1
            num_frames += 1

            boxes, confidences, classIDs = [], [], [] 
            vehicle_crossed_line_flag = False 

            start_time, num_frames = self.displayFPS(start_time, num_frames)
            (grabbed, frame) = videoStream.read()

            if not grabbed:
                break

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inputWidth, self.inputHeight),
                swapRB=True, crop=False)
            net_detect.setInput(blob)
            start = time.time()
            layerOutputs = net_detect.forward(ln)
            end = time.time()

            for output in layerOutputs:
                for i, detection in enumerate(output):
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.preDefinedConfidence:
                        box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                                    
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.preDefinedConfidence, self.preDefinedThreshold)

            self.drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

            vehicle_count, current_detections, vehicle_count_in_frame = self.count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)
            
            self.displayVehicleCount(frame, vehicle_count)

            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break	
            
            previous_frame_detections.pop(0)
            previous_frame_detections.append(current_detections)

            with open('{}{}.txt'.format(self.processedDataPath, data_name), 'a') as f:
                f.write(str(vehicle_count_in_frame)+'\n')
        
    def extract_startframe_and_endframe(self, previous_frame_detections, obj_pass, total_frame, data_name):
        temp = []
        for ob in obj_pass.keys():
            if ob not in previous_frame_detections[-1].values():
                with open('{}obj_startframe_and_endframe_{}.txt'.format(self.processedDataPath, data_name),'a') as f:
                    f.write('{} {} {}'.format(ob, obj_pass[ob], total_frame)+'\n')
                temp.append(ob)
        for i in temp:
            obj_pass.pop(i, None)
        temp.clear()
        for ob in previous_frame_detections[-1].values():
            if ob not in obj_pass.keys():
                obj_pass[ob] = total_frame
        return obj_pass

    def process_label(self, data_name, sourceVideofps):
        f = open('{}obj_startframe_and_endframe_{}.txt'.format(self.processedDataPath, data_name), 'r')
        data = f.read().split('\n')

        obj = {}
        for i in data[:-1]:
            k = i.split(' ')
            if int(k[0]) not in obj:
                obj[int(k[0])] = {}
                obj[int(k[0])]['start'] = int(k[1])
                obj[int(k[0])]['end'] = int(k[2])
            else:
                if int(k[2])>obj[int(k[0])]['end']:
                    obj[int(k[0])]['end'] = int(k[2])

        for i in obj:
            obj[i]['time'] = (obj[i]['end'] - obj[i]['start'])/29

        obj = {k: v for k, v in obj.items() if obj[k]['time'] > 4}

        label = []
        length = self.input_size
        last_key = list(obj)[-1]
        for k in range(obj[last_key]['end']):
            if k % length == 0:
                temp = []
                for j in range(1,11):
                    m = j*29
                    if m < obj[last_key]['end']:
                        s = []
                        for i in obj:
                            if obj[i]['start'] <= m and obj[i]['end'] >=m:
                                s.append(obj[i]['time'])
                        temp.append(np.mean(s))
                label.append(temp)
        
        for i in label:
            with open('{}label_{}.txt'.format(self.processedDataPath, data_name), 'a') as f:
                f.write(str(i)+'\n')
        return label



    def displayVehicleCount(self, frame, vehicle_count):
        cv2.putText(
            frame, #Image
            'Detected Vehicles: ' + str(vehicle_count), #Label
            (20, 20), #Position
            cv2.FONT_HERSHEY_SIMPLEX, #Font
            0.8, #Size
            (0, 0xFF, 0), #Color
            2, #Thickness
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            )

    def boxAndLineOverlap(self, x_mid_point, y_mid_point, line_scoordinates):
        x1_line, y1_line, x2_line, y2_line = line_coordinates

        if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
            (y_mid_point >= y1_line and y_mid_point <= y2_line+5):
            return True
        return False

    def displayFPS(self, start_time, num_frames):
        current_time = int(time.time())
        if(current_time > start_time):
            num_frames = 0
            start_time = current_time
        return start_time, num_frames

    def drawDetectionBoxes(self, idxs, boxes, classIDs, confidences, frame):
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

    def initializeVideoWriter(self, video_width, video_height, videoStream):
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter(self.outputVideoPath, fourcc, sourceVideofps,
            (video_width, video_height), True)

    def boxInPreviousFrames(self, previous_frame_detections, current_box, current_detections):
        centerX, centerY, width, height = current_box
        dist = np.inf 
        for i in range(self.FRAMES_BEFORE_CURRENT):
            coordinate_list = list(previous_frame_detections[i].keys())
            if len(coordinate_list) == 0: 
                continue
            temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
            if (temp_dist < dist):
                dist = temp_dist
                frame_num = i
                coord = coordinate_list[index[0]]

        if (dist > (max(width, height)/2)):
            return False

        current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
        return True

    def count_vehicles(self, idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
        current_detections = {}
        vehicle_count_in_frame = 0
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                centerX = x + (w//2)
                centerY = y+ (h//2)

                if (self.LABELS[classIDs[i]] in self.list_of_vehicles):
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count_in_frame += 1
                    if (not self.boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                        vehicle_count += 1

                    ID = current_detections.get((centerX, centerY))

                    if (list(current_detections.values()).count(ID) > 1):
                        current_detections[(centerX, centerY)] = vehicle_count
                        vehicle_count += 1 
                    cv2.putText(frame, str(ID), (centerX, centerY),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

        return vehicle_count, current_detections, vehicle_count_in_frame


    

        

    

    

