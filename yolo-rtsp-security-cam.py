import time
import os
import cv2
import queue
import threading
import numpy as np
from datetime import datetime
from ffmpeg import FFmpeg
from skimage.metrics import mean_squared_error as ssim
from sshkeyboard import listen_keyboard, stop_listening
from config import Config
cfg = Config()
cfg.load_from_file('config.yaml')

rtsp_stream = cfg.video

if cfg.frame_click:
    cfg.testing = True
    cfg.monitor = True
    print("frame_click enabled. Press any key to advance the frame by one, or hold down the key to advance faster. Make sure the video window is selected, not the terminal, when advancing frames.")
    
if len(cfg.yolo) > 0:
    yolo_list = cfg.yolo
    yolo_on = True
else:
    yolo_on = False

# Set up variables for YOLO detection
if yolo_on:
    from ultralytics import YOLO
    stop_error = False

    CONFIDENCE = 0.5

    labels = open("coco.names").read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    model = YOLO(cfg.model + ".pt")

    # Check if the user provided list has valid objects
    for coconame in yolo_list:
        if coconame not in labels:
            print("Error! '"+coconame+"' not found in coco.names")
            stop_error = True
    if stop_error:
        exit("Exiting")

# Set up other internal variables
loop = True
cap = cv2.VideoCapture(rtsp_stream)
fps = cap.get(cv2.CAP_PROP_FPS)
period = 1/fps
tail_length = cfg.tail_length * fps
recording = False
ffmpeg_copy = 0
activity_count = 0
yolo_count = 0
ret, img = cap.read()
if img.shape[1]/img.shape[0] > 1.55:
    res = (256,144)
else:
    res = (216,162)
blank = np.zeros((res[1],res[0]), np.uint8)
resized_frame = cv2.resize(img, res)
gray_frame = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
old_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
if cfg.monitor:
    cv2.namedWindow(rtsp_stream, cv2.WINDOW_NORMAL)

q = queue.Queue()
# Thread for receiving the stream's frames so they can be processed
def receive_frames():
    if cap.isOpened():
        ret, frame = cap.read()
    while ret and loop:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                q.put(frame)

# Record the stream when object is detected
def start_ffmpeg():
    try:
        ffmpeg_copy.execute()
    except:
        print("Issue recording the stream. Trying again.")
        time.sleep(1)
        ffmpeg_copy.execute()

# Functions for detecting key presses
def press(key):
    global loop
    if key == 'q':
        loop = False

def input_keyboard():
    listen_keyboard(
        on_press=press,
    )

def timer():
    delay = False
    period = 1
    now = datetime.now()
    now_time = now.time()
    start1 = now_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start2 = now_time.replace(hour=0, minute=0, second=1, microsecond=10000)
    start_t=time.time()
    while loop:
        now = datetime.now()
        now_time = now.time()
        if(now_time>=start1 and now_time<=start2):
            day_num = now.weekday()
            if day_num == 0: print("Monday "+now.strftime('%m-%d-%Y'))
            elif day_num == 1: print("Tuesday "+now.strftime('%m-%d-%Y'))
            elif day_num == 2: print("Wednesday "+now.strftime('%m-%d-%Y'))
            elif day_num == 3: print("Thursday "+now.strftime('%m-%d-%Y'))
            elif day_num == 4: print("Friday "+now.strftime('%m-%d-%Y'))
            elif day_num == 5: print("Saturday "+now.strftime('%m-%d-%Y'))
            elif day_num == 6: print("Sunday "+now.strftime('%m-%d-%Y'))
            delay = True
        time.sleep(period - ((time.time() - start_t) % period))
        if delay:
            delay = False
            time.sleep(period - ((time.time() - start_t) % period))

# Process YOLO object detection
def process_yolo():
    global img

    results = model.predict(img, conf=CONFIDENCE, verbose=False)[0]
    object_found = False

    # Loop over the detections
    for data in results.boxes.data.tolist():
        # Get the bounding box coordinates, confidence, and class id
        xmin, ymin, xmax, ymax, confidence, class_id = data

        # Converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        if labels[class_id] in yolo_list:
            object_found = True

        # Draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
        text = f"{labels[class_id]}: {confidence:.2f}"
        # Calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = img.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

        # Add opacity (transparency to the box)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Now put the text (label: confidence %)
        cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 0, 0), thickness=1)

    return object_found


# Start the background threads
receive_thread = threading.Thread(target=receive_frames)
receive_thread.start()
keyboard_thread = threading.Thread(target=input_keyboard)
keyboard_thread.start()
timer_thread = threading.Thread(target=timer)
timer_thread.start()

# Main loop
while loop:
    if q.empty() != True:
        img = q.get()

        # Resize image, make it grayscale, then blur it
        resized_frame = cv2.resize(img, res)
        gray_frame = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
        final_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)

        # Calculate difference between current and previous frame, then get ssim value
        diff = cv2.absdiff(final_frame, old_frame)
        result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
        ssim_val = int(ssim(result,blank))
        old_frame = final_frame

        # Print value for testing mode
        if cfg.testing and ssim_val > cfg.threshold:
            print("motion: "+ str(ssim_val))

        # Count the number of frames where the ssim value exceeds the threshold value.
        # If the number of these frames exceeds start_frames value, run YOLO detection.
        # Start recording if an object from the user provided list is detected
        if not recording:
            if ssim_val > cfg.threshold:
                activity_count += 1
                if activity_count >= cfg.start_frames:
                    if yolo_on:
                        if process_yolo():
                            yolo_count += 1
                        else:
                            yolo_count = 0
                    if not yolo_on or yolo_count > 1:
                        filedate = datetime.now().strftime('%H-%M-%S')
                        if not cfg.testing:
                            folderdate = datetime.now().strftime('%Y-%m-%d')
                            if not os.path.isdir(folderdate):
                                os.mkdir(folderdate)
                            filename = '%s/%s.mkv' % (folderdate,filedate)
                            ffmpeg_copy = (
                                FFmpeg()
                                .option("y")
                                .input(
                                    rtsp_stream,
                                    rtsp_transport="tcp",
                                    rtsp_flags="prefer_tcp",
                                )
                                .output(filename, vcodec="copy", acodec="copy")
                            )
                            ffmpeg_thread = threading.Thread(target=start_ffmpeg)
                            ffmpeg_thread.start()
                            print(filedate + " recording started")
                        else:
                            print(filedate + " recording started - Testing mode")
                        recording = True
                        activity_count = 0
                        yolo_count = 0
            else:
                activity_count = 0
                yolo_count = 0

        # If already recording, count the number of frames where there's no motion activity
        # or no object detected and stop recording if it exceeds the tail_length value
        else:
            if yolo_on and not process_yolo() or not yolo_on and ssim_val < cfg.threshold:
                activity_count += 1
                if activity_count >= tail_length:
                    filedate = datetime.now().strftime('%H-%M-%S')
                    if not cfg.testing:
                        ffmpeg_copy.terminate()
                        ffmpeg_thread.join()
                        ffmpeg_copy = 0
                        print(filedate + " recording stopped")
                        # If auto_delete argument was provided, delete recording if total
                        # length is equal to the tail_length value, indicating a false positive
                        if cfg.auto_delete:
                            recorded_file = cv2.VideoCapture(filename)
                            recorded_frames = recorded_file.get(cv2.CAP_PROP_FRAME_COUNT)
                            if recorded_frames < tail_length + (fps/2) and os.path.isfile(filename):
                                os.remove(filename)
                                print(filename + " was auto-deleted")
                    else:
                        print(filedate + " recording stopped - Testing mode")
                    recording = False
                    activity_count = 0
            else:
                activity_count = 0

        # Monitor the stream
        if cfg.monitor:
            cv2.imshow(rtsp_stream, img)
            if cfg.frame_click:
                cv_key = cv2.waitKey(0) & 0xFF
                if cv_key == ord("q"):
                    loop = False
                if cv_key == ord("n"):
                    continue
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    loop = False
    else:
        time.sleep(period/2)

# Gracefully end threads and exit
stop_listening()
if ffmpeg_copy:
    ffmpeg_copy.terminate()
    ffmpeg_thread.join()
receive_thread.join()
keyboard_thread.join()
timer_thread.join()
cv2.destroyAllWindows()
print("Exiting")
