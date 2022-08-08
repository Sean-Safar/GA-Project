#Completed By Sean Safar
#***** Actuator *****#
import RPi.GPIO as GPIO
#***** Camera *****#
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from skimage import exposure
import keyboard
#***** Load Cell *****#
from Phidget22.Devices.VoltageRatioInput import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

#***** GPIO Assignment for Actuator *****#
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(38,GPIO.OUT)
GPIO.setup(40,GPIO.OUT)
GPIO.setup(36,GPIO.OUT)
soft_pwm3 = GPIO.PWM(36,10)
#***** CAMERA PRE-PROCESSING Parameters *****#
camera = PiCamera()
frame_width = 640
frame_height = 480
camera.resolution = (frame_width, frame_height)
camera.framerate = 90
camera.sharpness = -40
camera.contrast = 22
camera.brightness = 46
camera.saturation = -28
camera.exposure_compensation = 9
camera.awb_mode = 'off'
gain_r = 1.7
gain_b = 1.2
camera.awb_gains = (gain_r, gain_b)
rawCapture = PiRGBArray(camera, size=(frame_width, frame_height))
#***** Ploting *****#
x_len = 8
y_range = [0,80]
fig_loadcell = plt.figure()
ax = fig_loadcell.add_subplot(1,1,1)
xs = list(range(0,8))
ys = [0] * x_len
ax.set_ylim(y_range)
line, = ax.plot(xs,ys)
plt.title('GA Pull Off Force')
plt.xlabel('Samples')
plt.ylabel('Force (N)')
#***** Load Cell *****#
r= [0]
output = []
ch = VoltageRatioInput()
ch.openWaitForAttachment(1000)
ch.setBridgeGain(BridgeGain.BRIDGE_GAIN_128)
ch.setDataInterval(8)
#***** Camera/Matrix Manipulation *****#
cameraMatrix = [[521.48889545,0,329.27954464],[0,532.21046939,237.99577431],[0,0,1]]
dist_param = [0.08646883,0.76458433,-0.00994538,-0.0233169,-1.20937861]
cameraMatrix = np.array(cameraMatrix)
dist_param = np.array(dist_param)
matrix_warp = [[1.10507573e+00,1.43890069e-02,-1.28608943e+01],[ 5.13105986e-02,1.01297053e+00,-5.01999724e+01],[2.60823434e-04,4.32063842e-05,1.00000000e+00]]
matrix_warp = np.array(matrix_warp)
width_warp, height_warp = 585,384
size_frame = (width_warp,height_warp)
file_name_vid = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/GA_PullOff1_" + str(time.time()) + ".avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
final_result = cv2.VideoWriter(file_name_vid,fourcc,1,size_frame)

#**********Get Contour Function**********#
def getContours(img,imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = 10000
        if area>areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255,0,255), 2)
            area = np.array(float(area))
            final_area = area / 143.441995464853
            cv2.putText(imgContour, "Area(mm2): {:.3f} ".format(float(final_area)), (0,350), cv2.FONT_HERSHEY_COMPLEX, 1,(0,69,255), 2)
            return(float(final_area))

#**********Animate function for Video and Load Cell**********#
def animate(i,output_,r_,ch,ys):
#***** Image Processing Code *****#
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        key = cv2.waitKey(1) & 0xFF
        image_norm = frame.array
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist_param, (frame_width,frame_height), 1, (frame_width,frame_height))
        image_undist = cv2.undistort(image_norm, cameraMatrix, dist_param, None, newCameraMatrix)
        image_warp = cv2.warpPerspective(image_undist,matrix_warp,(width_warp,height_warp))
        gamma_thres = 4
        imgGamma = exposure.adjust_gamma(image_warp,gamma_thres)
        pix_intensity = np.average(imgGamma)
        imgContour = imgGamma.copy()
        erode_thres = 6
        ker = np.ones((erode_thres,erode_thres), 'uint8')
        imgErode = cv2.erode(src= imgGamma,kernel=ker,iterations=1)
        imgGray = cv2.cvtColor(imgErode, cv2.COLOR_BGR2GRAY)
        gray_min_thres = 30
        gray_max_thres = 255
        _, imgGray = cv2.threshold(imgGray,gray_min_thres,gray_max_thres,cv2.THRESH_BINARY)
        canny_min_thres = 255
        canny_max_thers = 255
        imgCanny = cv2.Canny(imgGray,canny_min_thres,canny_max_thers)
        dilate_thres = 6
        kernel = np.ones((dilate_thres,dilate_thres), 'uint8')
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDil, imgContour)
        cv2.imshow("Contour", imgContour)
        final_result.write(imgContour)
        rawCapture.truncate(0)
        break
#***** LOAD CELL CODE *****#
    voltageRatio = ch.getVoltageRatio()
    if keyboard.is_pressed('t'):
        r_[0] = ch.getVoltageRatio()
        print("Tared")
    new_ratio = (voltageRatio - r_[0]) * 33291104.7583953
    new_ratio_F = new_ratio * .009806650028638
#***** Actuator Code *****#
    force_diff = 0
    if new_ratio_F == 0:
        GPIO.output(40,0)
        GPIO.output(38,1)
        soft_pwm3.start(10)
    if new_ratio_F >= 10 and new_ratio_F <= 18:
        soft_pwm3.ChangeDutyCycle(6)
    if new_ratio_F >= 60 and new_ratio_F <= 63:
        soft_pwm3.ChangeFrequency(6)
        soft_pwm3.ChangeDutyCycle(3)
    if new_ratio_F >= 8:
        force_diff = (float(output_[-2][0:4])) - (float(output_[-1][0:4]))
    if force_diff >= 20:
        GPIO.output(40,0)
        GPIO.output(38,0)
#***** Plot/Format*****#
    ys.append(new_ratio_F)
    ys = ys[-x_len:]
    line.set_ydata(ys)
    final_area = getContours(imgDil, imgContour)
    output_1 = "{},{},{}".format(new_ratio_F,final_area,pix_intensity)
    output_.append(output_1)
    return line,   

#**********Call to Animate Function**********#
ani = FuncAnimation(fig_loadcell, animate, fargs= [output,r,ch,ys],interval=1,blit=True)
plt.show()

#**********Extend Actuator**********#
soft_pwm3.ChangeFrequency(10)
soft_pwm3.ChangeDutyCycle(100)
GPIO.output(40,1)
GPIO.output(38,0)
time.sleep(3)
soft_pwm3.stop()
GPIO.cleanup()
final_result.release()

#**********Output Data to a CSV File**********#
header = ('Force (N)','Area (mm2)','Pixel Intensity (0-255)')
file_name_csv = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/Force_Area_PI1_" + str(time.time()) + ".csv"
with open(file_name_csv, 'w', encoding='UTF8') as f:
    f.write(",".join(header) + "\n")
    for x in output:
        f.write((str(x)) + "\n")