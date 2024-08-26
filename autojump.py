#! /usr/bin/env python
# -*- coding: utf-8
#
# autojump.py
# @Author Zahi
# @Time: 2024/8/251


import math
import time
import random
import subprocess
import logging

import cv2

class ADBUtils:
    @staticmethod
    def press(x: str | int | float, y: str | int | float, duration: str | int):
        command = f"adb shell input swipe {x} {y} {x} {y} {duration}"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)
    
    @staticmethod
    def click(x: str | int | float, y: str | int | float):
        command = f"adb shell input tap {x} {y}"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)
    
    @staticmethod
    def screenshot(filename: str):
        command = f"adb shell screencap -p /sdcard/{filename}.png"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)
    
    @staticmethod
    def pull(device: str, host: str):
        command = f"adb pull {device} {host}"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)
    
    @staticmethod
    def push(device: str, host: str):
        command = f"adb push {host} {device}"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)
    
    @staticmethod
    def mkdir(path: str):
        command = f"adb shell mkdir -p {path}"
        logging.info(f"Executing command: {command}")
        subprocess.run(command, shell=True)

class Detector:
    def __init__(self, template_img) -> None:
        # Load template image
        self.template = cv2.imread(template_img, cv2.IMREAD_GRAYSCALE)
        # Get template image size
        self.template_height, self.template_width = self.template.shape
        logging.info("Loaded template image")
    
    @staticmethod
    def is_grey(img):
        if len(img.shape) == 3:
            return False
        return True
    
    @staticmethod
    def convert_to_grayscale(img):
        if not Detector.is_grey(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __detect_object(self, img):
        logging.info("Detecting object...")
        img_gray = Detector.convert_to_grayscale(img)
        
        res = cv2.matchTemplate(img_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, _, _, start_coord = cv2.minMaxLoc(res)
        end_coord = (start_coord[0] + self.template_width, start_coord[1] + self.template_height)
        
        logging.info("Object detected: %s, %s", start_coord, end_coord)
        return (start_coord, end_coord)

    def __detect_contours(self, img):
        logging.info("Detecting contours...")

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        edge = cv2.Canny(blurred, 50, 150)

        # remove object
        start, end = self.__detect_object(img)
        edge[start[1]:end[1], start[0]:end[0]] = 0

        logging.info("Contours detected.")
        return edge

    def __detect_center_pos(self, edge_img):
        #! 索引为[y, x]
        start = (edge_img.shape[0] // 4, 0)
        end = (edge_img.shape[0] // 2, edge_img.shape[1])
        center = [0, 0]

        try:
            for i in range(start[0], end[0]):
                for j in range(start[1], end[1]):
                    if edge_img[i, j] == 255:
                        coord_a = (i, j)
                        center[0] = j
                        raise StopIteration
        except StopIteration:
            pass

        border_flag = False
        coord_b = []
        for i in range(coord_a[1], end[1]):
            border_flag = True
            for j in range(coord_a[0], end[0]):
                if edge_img[j, i] == 255:
                    if not border_flag:
                        coord_b.clear()
                    border_flag = False
                    coord_b.append(j)

            if border_flag:
                center[1] = sum(coord_b) // len(coord_b)
                break
        else:
            center[1] = sum(coord_b) // len(coord_b)

        logging.info("Center position detected: %s", center)
        return center

    def detect_distance(self, img):
        start, end = self.__detect_object(img)
        x1 = (start[0] + end[0]) // 2
        y1 = end[1]

        x2, y2 = self.__detect_center_pos(
            self.__detect_contours(img)
        )

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_duration(distance) -> int:
    return int(distance * 1.448) + random.randint(-10, 10)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    detector = Detector("obj.png")

    while True:
        logging.info("Start detecting...")

        ADBUtils.screenshot("screenshot")
        ADBUtils.pull("/sdcard/screenshot.png", "screenshot.png")

        img = cv2.imread("screenshot.png")
        
        distance = detector.detect_distance(img)
        logging.info("Distance: %s", distance)

        duration = calculate_duration(distance)
        logging.info("Duration: %s", duration)

        ADBUtils.press(random.randrange(0, 1000), random.randrange(600, 1500), duration)
        logging.info("Sleeping...")
        time.sleep(1)


def debug2(edge_img, start, end, center):
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # cv2.rectangle(edge_img, start, end, (255, 0, 0), 2)
    cv2.circle(edge_img, center, 10, (255, 0, 0), thickness=-1)
    cv2.imshow('1', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
