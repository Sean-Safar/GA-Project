#Completed By Sean Safar
#********************IMPORTS********************#
#***** Actuator *****#
import RPi.GPIO as GPIO                             #Raspberry Pi Software GPIO Access (There is a hardware PWM access option as well)

#***** Camera *****#
from picamera.array import PiRGBArray               #For Reading from the Camera (using PiRGBArray is suppose to help with processing time)
from picamera import PiCamera                       #For Preprocessing Adjustments (exposure,gain,sharpness,etc.)
import time                                         #For Time Date Stamp when Outputing Data
import cv2                                          #For OpenCV Filters
import numpy as np                                  #For Grabing Numpy Array of Raw Images & Array Manipulation in General Throughout Program
from skimage import exposure                        #For Gamma Filter
import keyboard                                     #For Interupting Animate Function to Initialize Tare/Actuation

#***** Load Cell *****#
from Phidget22.Devices.VoltageRatioInput import *   #For Phidget Bridge 1046_0 Connection (must run Sudo Command to Connect)
from matplotlib.animation import FuncAnimation      #Main Function Used in this Program (continuosly call animate function with paramters)
import matplotlib.pyplot as plt                     #For Plotting
#from itertools import count                        #Useful Library for Using Indexes (counter tool such as "index = count()")

#********************GLOBAL VARIABLES********************#
#***** GPIO Assignment for Actuator *****#
GPIO.setmode(GPIO.BOARD)                            #Board Mode Lets you Reference the GPIO Pins by Number
GPIO.setwarnings(False)                             #If not set Warnings will Show up for Accessing the GPIO Pins
GPIO.setup(38,GPIO.OUT)                             #PWM1 set to 1 for Retracting Direction otherwise 0
GPIO.setup(40,GPIO.OUT)                             #PWM2 set to 1 for Extending Direction otherwise 0
GPIO.setup(36,GPIO.OUT)                             #PWM3 set DC and Freq to control the Speed of Actuation
#soft_pwm1 = GPIO.PWM(38,10)                         
#soft_pwm2 = GPIO.PWM(40,10)
soft_pwm3 = GPIO.PWM(36,10)                         #Set PWM3 (control PWM on pin 36) to 10HZ


#***** CAMERA PRE-PROCESSING Parameters (use get_picam_paramters.py for adjustment) *****#
camera = PiCamera()                                 #Connect to PiCamera
frame_width = 640                                   #1280 #1920 #2592 
frame_height = 480                                  #720  #1080 #1944 
camera.resolution = (frame_width, frame_height)     #Other Resolutions Above
camera.framerate = 90                               #Max 90 FPS w/OV5647 @ 640:480
camera.sharpness = -40                              #-100:to:100
camera.contrast = 22                                #-100:to:100
camera.brightness = 46                              #   0:to:100
camera.saturation = -28                             #-100:to:100
camera.exposure_compensation = 9                    #-100:to:100
camera.awb_mode = 'off'                             #A lot of Auto Modes see get_picam_parameters.py
gain_r = 1.7                                        #   0:to:8
gain_b = 1.2                                        #   0:to:8
camera.awb_gains = (gain_r, gain_b)
rawCapture = PiRGBArray(camera, size=(frame_width, frame_height))    #Array for Reading Camera Frames
#time.sleep(1)                                      #Allow Camera to "Warmup"
#h,  w = image_norm.shape[:2]                       #May be Useful for further implementation (syntac to get current height and width of image and set to variables)

#***** Ploting *****#
x_len = 8                                           #Number of points to display
y_range = [0,80]                                    #Range of possible y vals
fig_loadcell = plt.figure()                         #Create figure for plotting
ax = fig_loadcell.add_subplot(1,1,1)                #SubPlot for Load Cell (can add additionaly plots)
xs = list(range(0,8))                               #Only Display 8 Values at a Time on Plot (Make smaller for faster program)
ys = [0] * x_len                                    #Zero Array for Range
ax.set_ylim(y_range)                                #Set Limits
line, = ax.plot(xs,ys)                              #Line that is Plotted
plt.title('GA Pull Off Force')                      #Formatting
plt.xlabel('Samples')
plt.ylabel('Force (N)')

#***** Load Cell *****#
r= [0]                                              #Tare Value Passed into Animate Function
output = []                                         #Main Output List to CSV File
ch = VoltageRatioInput()                            #Set up connection to Load Cell
ch.openWaitForAttachment(1000)                      #Wait for Load Cell to Connect
ch.setBridgeGain(BridgeGain.BRIDGE_GAIN_128)        #Gain Relates to Resolution of Load Cell Reading (128 is max)
ch.setDataInterval(8)                               #Sets Data Rate (8 is fastest)

#***** Camera Matrix Manipulation *****#
cameraMatrix = [[521.48889545,0,329.27954464],                          #Camera Matrix for undistortion (output from cam_calibration.py)
                [0,532.21046939,237.99577431],
                [0,0,1]] 
dist_param = [0.08646883,0.76458433,-0.00994538,-0.0233169,-1.20937861] #Distortion Vector (output from cam_calibration.py)
cameraMatrix = np.array(cameraMatrix)                                   #Must Convert to numpy array
dist_param = np.array(dist_param)                                       #Must Convert to numpy array

matrix_warp = [[1.10507573e+00,1.43890069e-02,-1.28608943e+01],         #Warping Matric (for eagle eye view, get values from code in Animate Function)
               [ 5.13105986e-02,1.01297053e+00,-5.01999724e+01],
               [2.60823434e-04,4.32063842e-05,1.00000000e+00]]
matrix_warp = np.array(matrix_warp)                                     #Must Convert to numpy array
width_warp, height_warp = 585,384                                       #Can be any integer value (chose 384 because pixel size of actual undistorted Rectangle with lowest amount of distortion was 384)(keep aspect ratio by 384*1.523645=582)

size_frame = (width_warp,height_warp)                                                                           #For writing to video file
file_name_vid = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/GA_PullOff1_" + str(time.time()) + ".avi"  #Output Video Format/Location (look into lossless compression)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')                                                                        #Tried *'X264' as well, MJPG seems the best with loss compression
final_result = cv2.VideoWriter(file_name_vid,fourcc,1,size_frame)                                               #Change to 3 for real time video (OV5647) (with new camera sensor keep at 1 or look into using 0 with lossless compression)

#**********Creating Gui Slider Bars**********#
# def empty(a):
#     pass
# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 420,100)
# #cv2.createTrackbar("clip_limit", "Parameters",1,20,empty)        #For Enhancement Filter (removed for time optimization) (allows better display of wedge features)(.5 was good)
# #cv2.createTrackbar("tile_grid_size", "Parameters",1,100,empty)   #For Enhancement Filter (removed for time optimization) (allows better display of wedge features use 100)(37 was good)
# #cv2.createTrackbar("Min ImgCanny", "Parameters",0,255,empty)     #For Canny Filter (Min black/white threshold)
# #cv2.createTrackbar("Max ImgCanny", "Parameters",0,255,empty)     #For Canny Filter (Max black/white threshold)
# cv2.createTrackbar("Min ImgGray", "Parameters",0,255,empty)       #For Gray Filter (Min black/white threshold)
# #cv2.createTrackbar("Max ImgGray", "Parameters",0,255,empty)      #For Gray Filter (Maz black/white threshold)
# cv2.createTrackbar("Dilate", "Parameters",0,30,empty)             #Dilation Threshold (increase for increase dilation size)
# cv2.createTrackbar("Erode", "Parameters",0,30,empty)              #Eroding Threshold (increase for increase eroding size) keep the same as dilation or will mess with area calculation
# cv2.createTrackbar("Gamma", "Parameters",0,20,empty)              #Gamma Threshold
# cv2.createTrackbar("Area", "Parameters",1000,40000,empty)         #Area Filter (sets the min accepted area value, anythiung below will not be drawn or caluclated)

#**********Stack Images**********#
#For Comparing All Filters to the Image (uncomment Gui Slider Bars for adjustment of parameters (some filters must have at least 1 as value to display image))
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width_stack = imgArray[0][0].shape[1]
    height_stack = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height_stack, width_stack, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#**********Get Contour Function**********#
def getContours(img,imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Change RETR for different contours

    for cnt in contours:
        area = cv2.contourArea(cnt)                                 #Utilizes Greens theorem to compute area of Contour
        #areaMin = cv2.getTrackbarPos("Area", "Parameters")         #Uncomment for GUI
        areaMin = 10000                                             #Min Area Threshold 10000 pixels comment out for GUI
        if area>areaMin:                                            
            cv2.drawContours(imgContour, cnt, -1, (255,0,255), 2)   #Draw Contour Function (-1 = draw all contours)(255,0,255 = color of contour)(2 = line width)
            
            #peri = cv2.arcLength(cnt, True)                        #Uncomment and uncomment below to show a bounding rectange around contour (can be utilized with pixel calculation for increased accuracy)
            #approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            #x,y,w,h = cv2.boundingRect(approx)
            #cv2.rectangle(imgContour, (x-20, y-20), (x + w+20, y + h+20), (0,0,0),3)
            #cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w -240, y + 300), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3)  #Used to display perimeter of polydp points used for bounding rectangle
            
            area = np.array(float(area))                            #Must Conevert Area to float and then Numpy
            final_area = area / 143.441995464853                    #143.4 is from Pixel Per Metric calibration with black 3x3cm FTIR square (make sure to use Pixel Per Metric Ratio with all filters applied to there threshold values for accurate readings)
            cv2.putText(imgContour, "Area(mm2): {:.3f} ".format(float(final_area)), (0,350), cv2.FONT_HERSHEY_COMPLEX, 1,(0,69,255), 2)     #Formatting Area Display 3 decimal points, (0,350) = Arrangement of Text on Screen, 1 = text size, (0,69,255) = color of text, 2 = width of text
            
            return(float(final_area))                               #Have to return area for outputing to CSV

#**********Animate function for Video and Load Cell**********#
def animate(i,output_,r_,ch,ys):    #Input i=total data count or i for undetermined, output_ = output list for csv, r_ = tare value, ch = Load Cell Channel, ys = ys axis have to pass in for Plotting feature

 #***** Image Processing Code *****#
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):          #Grab the raw NumPy array representing the image
        key = cv2.waitKey(1) & 0xFF                                                                 #Need this line for sudo command to run with Video
        image_norm = frame.array                                                                    #Normal Image
       
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist_param, (frame_width,frame_height), 1, (frame_width,frame_height)) #Get new Camera Matrix (need roi or get buffer error)
        image_undist = cv2.undistort(image_norm, cameraMatrix, dist_param, None, newCameraMatrix)   #Undistortion Matrix Application
        
    #*****Uncommment to get new warp_matrix*****#
        #pts1 = np.float32([[11,49],[623,18],[6,433],[628,463]])                                    #Set these equal to the four points of Pixel Loactions found on Micorsoft Paint of Calibration Image (used black rectangle, pts Order go top left, top right, bottom left, bottom right)
        #pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])                             #Order Matters here, as long as follow above example should work
        #cv2.circle(image_undist,(int(pts1[0][0]),int(pts1[0][1])),5,(0,255,0),-1)                  #Warp Point Set-up without Forloop
        #cv2.circle(image_undist,(int(pts1[1][0]),int(pts1[1][1])),5,(0,255,0),-1)
        #cv2.circle(image_undist,(int(pts1[2][0]),int(pts1[2][1])),5,(0,255,0),-1)
        #cv2.circle(image_undist,(int(pts1[3][0]),int(pts1[3][1])),5,(0,255,0),-1)
        #matrix_warp = cv2.getPerspectiveTransform(pts1,pts2)                                       #Outputs Warp Matric
        #print(matrix_warp)                                                                         #Print to get value for Global Variable

        image_warp = cv2.warpPerspective(image_undist,matrix_warp,(width_warp,height_warp))         #Warp Persepective for Eagle Eye View
        #gamma_thres = cv2.getTrackbarPos("Gamma", "Parameters")
        gamma_thres = 4                                                                             #Increase for Less Noise, vice versa
        imgGamma = exposure.adjust_gamma(image_warp,gamma_thres)                                    #Apply Gamme
        pix_intensity = np.average(imgGamma)                                                        #can add ",axis(0,1)" for all channels intensity (RGB), can change for bounding rectangle for more accurate but also loss of frames

    #*****Uncomment to use enhancement filter*****#
        #lab = cv2.cvtColor(imgGamma, cv2.COLOR_BGR2LAB)
        #l_channel, a, b = cv2.split(lab)
        #clip_limit_thres = cv2.getTrackbarPos("clip_limit", "Parameters")          #Uncomment for GUI
        #tile_grid_size_thres = cv2.getTrackbarPos("tile_grid_size", "Parameters")  #Uncomment for GUI
        #tile_grid_size_thres = 37                                                  #increase to around 100 if desired to see wedges
        #clahe = cv2.createCLAHE(clipLimit=clip_limit_thres, tileGridSize=(tile_grid_size_thres,tile_grid_size_thres))
        #cl = clahe.apply(l_channel)                                                #Applying CLAHE to L-channel                                                 
        #limg = cv2.merge((cl,a,b))                                                 #Merge the CLAHE enhanced L-channel with the a and b channel
        #enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)                       #Converting image from LAB Color model to BGR color space

        imgContour = imgGamma.copy()                                                #Image to display Contour and Area on (can use any)
        #erode_thres = cv2.getTrackbarPos("Erode", "Parameters")                    #Uncomment for GUI
        erode_thres = 6                                                             #6 Recieved good results (can decrease with cleaner glass/acrylic)
        ker = np.ones((erode_thres,erode_thres), 'uint8')                           #Kernel Size for Eroding (can alter kernel type as well)
        imgErode = cv2.erode(src= imgGamma,kernel=ker,iterations=1)                 #Erode Filter (do not increase iterations has direct correlation to FPS)
        imgGray = cv2.cvtColor(imgErode, cv2.COLOR_BGR2GRAY)                        #Gray Scale Filter
        #gray_min_thres = cv2.getTrackbarPos("Min ImgGray", "Parameters")           #Uncomment GUI
        #gray_max_thres = cv2.getTrackbarPos("Max ImgGray", "Parameters")           #Uncomment GUI
        gray_min_thres = 30                                                         #Min Value (this value offers the most control over entire contour so very imortant)
        gray_max_thres = 255                                                        #Maxed to 255 (lowering most likely will not change anything)
        _, imgGray = cv2.threshold(imgGray,gray_min_thres,gray_max_thres,cv2.THRESH_BINARY)
        #canny_min_thres = cv2.getTrackbarPos("Min ImgCanny", "Parameters")         #Uncomment GUI
        #canny_max_thres = cv2.getTrackbarPos("Max ImgCanny", "Parameters")         #Uncomment GUI
        canny_min_thres = 255                                                       #Maxed to simplify Contour
        canny_max_thers = 255                                                       #Maxed to simplify Contour (both canny adjustments have little affect under FTIR
        imgCanny = cv2.Canny(imgGray,canny_min_thres,canny_max_thers)
        #dilate_thres = cv2.getTrackbarPos("Dilate", "Parameters")              
        dilate_thres = 6                                                            #Same as Erode for Area Consistency Reasons
        kernel = np.ones((dilate_thres,dilate_thres), 'uint8')                      #Kernel Type for Dilate (can alter kernel type)
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)                         #Dilate Filter (do not increase iterations has direct correlation to FPS)
        getContours(imgDil, imgContour)                                             #Send final image to "getContour" Function
        
        #imgStack = stackImages(.4,([image_norm,image_warp,imgGamma,enhanced_img],[imgErode,imgGray,imgDil,imgContour]))    #Uncomment for all Preview Windows of Filters, inorder of application to the image frame (can switc out to show undistorted,enhanced,etc.)
        #cv2.imshow("Normal   Undist   Gamma   Enhanced   ****   Erode   Gray   Dilate   Contour", imgStack)                #Uncomment to show the stacked images with correct graph name, left to right top to bottom
    
        cv2.imshow("Contour", imgContour)                                           #Show Contour Image
        final_result.write(imgContour)                                              #Write Video Frame to Video Writer
    
        rawCapture.truncate(0)                                                      #clear the stream in preparation for the next frame
        break                                                                       #comment out to only show video stream (no load cell)
        #rawCapture.seek(0)                                                         #helps with video clear stream (may not be needed)

#*****LOAD CELL CODE*****#
    voltageRatio = ch.getVoltageRatio()                     #Voltage Value from Load Cell
    if keyboard.is_pressed('t'):                            #Hit "t" key to Tare Load Cell and Start Actuator
        r_[0] = ch.getVoltageRatio()                        #Tare Value
        print("Tared")
    new_ratio = (voltageRatio - r_[0]) * 33291104.7583953   #33291104 Recieved from Calibration Process
    new_ratio_F = new_ratio * .009806650028638              #Convert Units to Newtons

#*****Actuator Code*****#
    force_diff = 0                                  #Have to Initiliaze
    if new_ratio_F == 0:                            #When Load Cell Tared, begin actuation (can add a wait here if desired)
        GPIO.output(40,0)
        GPIO.output(38,1)
        soft_pwm3.start(10)
    if new_ratio_F >= 10 and new_ratio_F <= 18:     #If Between these Values (cannot use just one value because of slower data) Change DC to decrease Retract Rate
        #soft_pwm3.ChangeFrequency(10)
        soft_pwm3.ChangeDutyCycle(6)
    if new_ratio_F >= 60 and new_ratio_F <= 63:     #If Between these Values Change DC and Freq to decrease Retract Rate and Force
        soft_pwm3.ChangeFrequency(6)
        soft_pwm3.ChangeDutyCycle(3)
    if new_ratio_F >= 8:                                                    #Any Number After 0 so that the List can Populate before calculating the difference seen below
        force_diff = (float(output_[-2][0:4])) - (float(output_[-1][0:4]))  #Compute Difference between Current Force of Output_,(Colummn 1,digits 0-4), and Previous Value of Force of Output_,(Colummn 1,digits 0-4)
        #print(float(output_[-2][0:4]))
        #print(float(output_[-1][0:4]))
        #print(force_diff)
    if force_diff >= 20:                                                    #Can set 20 = to anyvalue, this sets the threshold to turn actuator off after pull-off with above difference caluclation
        GPIO.output(40,0)                                                   #Retract Off
        GPIO.output(38,0)                                                   #Extend Off

#*****Plot/Format*****#
    ys.append(new_ratio_F)                                                  #Add y to list
    ys = ys[-x_len:]                                                        #Limit y list to set number of items
    line.set_ydata(ys)                                                      #Update line with new y values
    final_area = getContours(imgDil, imgContour)                            #Returns Final Area of Current Frame
    output_1 = "{},{},{}".format(new_ratio_F,final_area,pix_intensity)      #Output on this format to remove unwanted brackets or commas
    output_.append(output_1)                                                #Output to List to Later be sent to CSV

    return line,                                                            #Return Line for Plotting

#*****************************************#         

#**********Call to Animate Function**********#
ani = FuncAnimation(fig_loadcell, animate, fargs= [output,r,ch,ys],interval=1,blit=True) #Interval set to lowest, blit control cache
plt.show()

#**********Extend Actuator**********#
soft_pwm3.ChangeFrequency(10)   #Change Speed to Faster
soft_pwm3.ChangeDutyCycle(100)  #Change Speed to Faster
GPIO.output(40,1)               #Extend On
GPIO.output(38,0)               #Retract Off
time.sleep(3)                   #Wait for Full Extention of Actuator
soft_pwm3.stop()                #Stop Control Signal
GPIO.cleanup()                  #Clear Signals
final_result.release()          #Release Video Writer (not essential)

#**********Output Data to a CSV File**********#
header = ('Force (N)','Area (mm2)','Pixel Intensity (0-255)')
file_name_csv = "/media/flexiv-user/LINUXCNC 2_/Actuator Integration/Force_Area_PI1_" + str(time.time()) + ".csv"
#The Below Code was commented out but this method may need to be used if faster total pull-off actuation test time is desried
#Deleting output[0:2][:] (zero values for actuator actuation to run properly)
#del output[0:2]
with open(file_name_csv, 'w', encoding='UTF8') as f:
    f.write(",".join(header) + "\n")
    for x in output:
        f.write((str(x)) + "\n")
