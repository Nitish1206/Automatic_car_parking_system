import json
import math
import threading
import numpy as np
import cv2
# import paho.mqtt.client as paho
import tensorflow as tf
from scipy.spatial import distance as dist
from tracker import *
from get_license_number import get_result_api
# Create tracker object
from threading import Thread
import time
# https://github.com/JunshengFu/vehicle-detection
# https://github.com/cw1204772/AIC2018_iamai
# https://www.youtube.com/watch?v=A8BHiuDvmRY
# https://www.youtube.com/watch?v=FdZvMoP0dRU
import datetime
from flask import Flask,jsonify
from flask_cors import CORS, cross_origin
from park_status import get_status

app = Flask(__name__)
CORS(app, support_credentials=True)

global data
data = []

class Parking_allocation():

    #initialize
    def __init__(self):
        self.rect_points = []
        self.two_consecutive_points = []
        self.COLORS = np.random.uniform(0, 255, size=(1, 3))
        self.checkin_status = False
        self.parking_status = False
        self.checkout_status = False
        self.print_checkin_status = False
        self.print_parking_status = False
        self.print_checkout_status = False

        self.status_data = {}
        self.print_status = {}
        self.colour_dict = {}
        self.park_route_up = {}
        self.park_route_down = {}
        self.park_route_center = {}
        self.car_in_cord = (0, 0)
        self.car_park_cord = (0, 0)
        self.car_out_cord = (0, 0)
        self.tracker = EuclideanDistTracker()
        self.number_of_parking = 3
        self._dict = {}
        self.time_dict={}
        self.park_status={}
        self.park_out_count={}
        self.park_in_count={}

    def main(self):

        parking_area = []

        with tf.gfile.FastGFile('final_model/frozen_inference_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Session() as self.sess:
            # Restore session
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            cap = cv2.VideoCapture("N:\Projects\Vehicle_update\\videos\\test3.mp4")
            counter = 0
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            size = (frame_width, frame_height)
            # result_vid = cv2.VideoWriter('testing1.avi',
                                         # cv2.VideoWriter_fourcc(*'MJPG'),
                                         # 30, size)
            frame_counter = 0
            pa_name = 0
            global data

            data = []
            while True:
                _, self.image = cap.read()
                current_frame=self.image.copy()
                frame_counter += 1
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.on_mouse)
                center_of_line = []

                for rect_p in self.rect_points:
                    cv2.circle(self.image, tuple(rect_p), 2, (255, 0, 0), -1)

                if len(self.rect_points) == 4:
                    pa_name += 1
                    center_points = []
                    x = [p[0] for p in self.rect_points]
                    y = [p[1] for p in self.rect_points]
                    ordered_points = self.order_points(np.array(self.rect_points))
                    for k in range(4):
                        if k < 3:
                            j = k+1
                        if k == 3:
                            j = 0

                        centr_p = self.get_center(self.rect_points[k], self.rect_points[j])
                        center_points.append(centr_p)

                    ys_centr = [p[1] for p in center_points]
                    max_ys = max(ys_centr)
                    min_ys = min(ys_centr)

                    top_center = center_points[ys_centr.index(min_ys)]
                    bottom_center = center_points[ys_centr.index(max_ys)]
                    centroid = (sum(x) // len(self.rect_points), sum(y) // len(self.rect_points))
                    all_dist=[]
                    for cp in center_points:
                        all_dist.append(self.get_distance(centroid,cp,"xy"))

                    # dist_in = self.get_distance(centroid, top_center, "xy")
                    dist_in = min(all_dist)
                    dist_out = self.get_distance(centroid, bottom_center, "xy")
                    pts_list = np.array(self.rect_points, np.int32)
                    pts_list = pts_list.reshape((-1, 1, 2))
                    parking_area.append([centroid, dist_in, dist_out, top_center, bottom_center, [pts_list],
                                         pa_name,ordered_points])

                    self.rect_points = []

                if len(parking_area) > 0:
                    self.image, car_rects = self.main_process(self.image)

                    p = 0
                    for pa in parking_area:

                        p += 1
                        parking_name = "parking  " + str(p)

                        if parking_name not in self.print_status.keys():
                            self.print_status[parking_name] = [True, True, True, 0]
                            self.colour_dict[parking_name] = [(0, 0, 255), (0, 0, 255), (0, 0, 255)]
                            self.time_dict[parking_name]=0
                            self.park_status[parking_name]=False
                            self.park_route_up[parking_name] = False
                            self.park_route_down[parking_name] = False
                            self.park_out_count[parking_name]=0
                            self.park_in_count[parking_name]=0
                            park_data = {"id": p, "color": "red", "text": ""}
                            data.append(park_data)

                        cv2.polylines(self.image, pa[5], True, (255, 0, 0))

                        car_dist_to_top_dist = []
                        car_dist_to_park_dist = []
                        car_dist_to_bottom_dist = []

                        car_dist_to_top_dist_c = []
                        car_dist_to_park_dist_c = []
                        car_dist_to_bottom_dist_c = []
                        all_rect = []

                        for car_rect in car_rects:
                            all_rect.append(car_rect)
                            distance_t, car_ct = self.near_parking(car_rect[0], car_rect[1], car_rect[2], car_rect[3],
                                                                  pa[3])
                            car_dist_to_top_dist.append(distance_t)
                            car_dist_to_top_dist_c.append(car_ct)

                            distance_c, car_cc = self.near_parking(car_rect[0], car_rect[1], car_rect[2], car_rect[3],
                                                                  pa[0])
                            car_dist_to_park_dist.append(distance_c)
                            car_dist_to_park_dist_c.append(car_cc)

                            distance_b, car_cb = self.near_parking(car_rect[0], car_rect[1], car_rect[2], car_rect[3],
                                                                  pa[4])
                            car_dist_to_bottom_dist.append(distance_b)
                            car_dist_to_bottom_dist_c.append(car_cb)

                        # if len(car_dist_to_top_dist) > 0:
                        #     if min(car_dist_to_top_dist) < pa[1]:
                        #         nearest_car_ct = car_dist_to_top_dist_c[car_dist_to_top_dist.index(min(car_dist_to_top_dist))]
                        #         if nearest_car_ct[1] > pa[3][1] and not self.park_route_down[parking_name]:
                        #             self.park_route_up[parking_name] = True
                                    # if nearest_car_ct[1] < pa[3][1]:
                                    #     self.park_route[parking_name] = [False, True]
                                    # if self.print_status[parking_name][0]:
                                        # self.print_status[parking_name][0] = False
                                        # self.print_status[parking_name][2] = True
                                        # self.print_status[parking_name][1] = True
                                    # print("car arriving to ==>>", parking_name)
                                # if nearest_car_ct[1] < pa[3][1] and self.park_route_down[parking_name]:

                                    # if self.print_status[parking_name][2]:
                                    #     self.colour_dict[parking_name][1] = (0, 0, 255)
                                    #     self.print_status[parking_name][2] = False
                                    #     self.print_status[parking_name][0] = True
                                    #     park_time = time.time()-self.time_dict[parking_name]
                                    #     print("Car Leaving From ==>>", parking_name)
                                    #     park_data = {"id": p, "color": "red", "text": "Thank you your park time:"+str(park_time)}
                                    #     data[p - 1] = park_data
                        # if len(car_dist_to_bottom_dist) > 0:
                        #     if min(car_dist_to_bottom_dist) < pa[1]:
                        #         nearest_car_cb = car_dist_to_bottom_dist_c[car_dist_to_bottom_dist.index(min(car_dist_to_bottom_dist))]
                        #         if nearest_car_cb[1] < pa[4][1]:
                        #             self.park_route_down[parking_name] = True
                        #
                        #         if nearest_car_cb[1] > pa[4][1] and self.park_route_up:
                        #
                        #             if self.print_status[parking_name][2]:
                        #                 self.colour_dict[parking_name][1] = (0, 0, 255)
                        #                 self.print_status[parking_name][2] = False
                        #                 self.print_status[parking_name][0] = True
                        #                 park_time = time.time()-self.time_dict[parking_name]
                        #                 print("Car Leaving From ==>>", parking_name)
                        #                 park_data = {"id": p, "color": "red", "text": "Thank you your park time:"+str(park_time)}
                        #                 data[p - 1] = park_data

                                # if nearest_car_cb[1] < pa[4][1]:
                                #     if self.print_status[parking_name][0]:
                                #         self.print_status[parking_name][0] = False
                                #         self.print_status[parking_name][2] = True
                                #         self.print_status[parking_name][1] = True
                                #         print("car arriving to ==>>", parking_name)

                        if len(car_dist_to_park_dist) > 0:
                            if min(car_dist_to_park_dist) < pa[1] and not self.park_status[parking_name]:
                                car_index = car_dist_to_park_dist.index(min(car_dist_to_park_dist))
                                nearest_car_cc = car_dist_to_park_dist_c[car_index]

                                park_status=get_status(pa[7],nearest_car_cc)
                                if park_status:
                                    self.park_in_count[parking_name] += 1
                                    if self.park_in_count[parking_name] >= 4:
                                        self.park_in_count[parking_name]=0
                                        self.park_status[parking_name]=True
                                        self.colour_dict[parking_name][1] = (0, 255, 0)
                                        self.time_dict[parking_name] = datetime.datetime.now()
                                        car_rect = all_rect[car_index]
                                        car_image = self.image[car_rect[1]:car_rect[3], car_rect[0]:car_rect[2]]
                                        number_plate = get_result_api(car_image)

                                        if number_plate != None:
                                            park_data = {"id": p, "color": "green", "text": "occupied by " + str(number_plate)}
                                            data[p-1] = park_data
                                        if number_plate == None:
                                            park_data = {"id": p, "color": "green", "text": "occupied by plate not recognized"}
                                            data[p-1] = park_data

                            if self.park_status[parking_name]:
                                car_index = car_dist_to_park_dist.index(min(car_dist_to_park_dist))
                                nearest_car_cc_ = car_dist_to_park_dist_c[car_index]

                                park_status_ = get_status(pa[7], nearest_car_cc_)
                                if not park_status_:
                                    self.park_out_count[parking_name] += 1
                                    if self.park_out_count[parking_name] >= 4:
                                        self.park_out_count[parking_name]=0

                                        self.colour_dict[parking_name][1] = (0,0,255)
                                        park_time_ = (datetime.datetime.now()-self.time_dict[parking_name])
                                        park_time=divmod(park_time_.total_seconds(), 60)
                                        park_data = {"id": p, "color": "red", "text": " Thank you! your parking time:"+
                                                     str(park_time[0])+"minutes"+str(park_time[1])+"seconds"}
                                        data[p - 1] = park_data
                                        self.park_status[parking_name] = False

                                # if self.park_route_up[parking_name]:
                                #     if nearest_car_cc[1] > pa[0][1]:
                                #
                                #         car_rect = all_rect[car_index]
                                #         car_image = self.image[car_rect[1]:car_rect[3], car_rect[0]:car_rect[2]]
                                #         number_plate = get_result_api(car_image)
                                #
                                #         if number_plate != None:
                                #
                                #             self.colour_dict[parking_name][1] = (0, 255, 0)
                                #             self.print_status[parking_name][1] = False
                                #             self.time_dict[parking_name]=time.time()
                                #             print("car parked at ==>>", parking_name)
                                #             print("car license ==>>", number_plate)
                                #
                                #             park_data = {"id": p, "color": "green", "text": "occupied by "+str(number_plate)}
                                #             data[p-1] = park_data
                                # if self.park_route_down[parking_name]:
                                #     if nearest_car_cc[1] < pa[0][1]:
                                #
                                #         car_rect = all_rect[car_index]
                                #         car_image = self.image[car_rect[1]:car_rect[3], car_rect[0]:car_rect[2]]
                                #         number_plate = get_result_api(car_image)
                                #
                                #         if number_plate != None:
                                #             self.colour_dict[parking_name][1] = (0, 255, 0)
                                #             self.print_status[parking_name][1] = False
                                #             self.time_dict[parking_name] = time.time()
                                #             print("car parked at ==>>", parking_name)
                                #             print("car license ==>>", number_plate)
                                #
                                #             park_data = {"id": p, "color": "green",
                                #                          "text": "occupied by " + str(number_plate)}
                                #             data[p - 1] = park_data

                        cv2.circle(self.image, tuple(pa[0]), 5, self.colour_dict[parking_name][1], -1)

                cv2.imshow("image", self.image)

                key = cv2.waitKey(1)

                if key == ord('q'):
                    break

            cap.release()
            # result_vid.release()
            cv2.destroyAllWindows()

    def get_distance(self,point1,point2,axis):
        if axis == "x":
            return  math.sqrt((point1[0]-point2[0])**2)
        if axis == "y":
            return math.sqrt((point1[1] - point2[1]) ** 2)
        if axis == "xy":
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def near_parking(self,startX, startY, endX, endY, parking_point):
        tempX = int((startX+endX)/2)
        tempY = int((startY+endY)/2)
        car_height = endY-startY
        car_center = (tempX, tempY+(car_height//2))
        cv2.circle(self.image, tuple(car_center), 5, (255, 0, 0), -1)
        dist_xy = self.get_distance(parking_point, car_center, "xy")
        return dist_xy, car_center

    def main_process(self, image):
        car_rects=[]
        rows = image.shape[0]
        cols = image.shape[1]
        inp = cv2.resize(image, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # Run the model
        out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
                        self.sess.graph.get_tensor_by_name('detection_scores:0'),
                        self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                        self.sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])

        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.6 and classId == 3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                y = int(y) - 15 if int(y) - 15 > 15 else int(y) + 15

                right = bbox[3] * cols
                bottom = bbox[2] * rows
                label = "{}".format("car")
                cv2.rectangle(image, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                car_rects.append([int(x), int(y), int(right), int(bottom)])
                cv2.putText(image, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[0], 2)

        boxes_ids = self.tracker.update(car_rects)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(image, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return image, car_rects

    def on_mouse(self, event, x, y, flags, params):

        # get mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_points.append([x, y])
            self.two_consecutive_points.append([x,y])

    def get_center(self,point1, point2):
        return [(point1[0]+point2[0])//2, (point1[1]+point2[1])//2]

    def order_points(self,pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return [tl, tr, br, bl]


@app.route("/")
@cross_origin(supports_credentials=True)
def send_data():
    return jsonify(data)


if __name__ == '__main__' :
    allocation_obj = Parking_allocation()
    main_treadd=Thread(target=allocation_obj.main)
    main_treadd.start()
    app.run()


