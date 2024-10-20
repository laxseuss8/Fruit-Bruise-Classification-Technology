import RPi.GPIO as gp
import os
import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
import libcamera
from libcamera import controls
import time
import csv
import joblib
from skimage.measure import label, regionprops
from tkinter import *
from PIL import Image as PILImage, ImageTk
import PIL.Image

# Set up GPIO warnings and mode
gp.setwarnings(False)
gp.setmode(gp.BOARD)

# Set up GPIO pins
gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

# Define directories for raw and processed images
RAW_IMAGE_DIR = "images"
PROCESSED_IMAGE_DIR = "processed"

# Create directories if they don't exist
os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

CSV_FILE = "features.csv"
MODEL = r"/home/bellpepper/RaspberryPi/Model/bagging_classifier.pkl"


def load_model():
    return joblib.load(MODEL)


# Load the model once at the beginning
model = load_model()

# Prepare the CSV file if it doesn't exist
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [
            'Mean_Area', 'Mean_Perimeter', 'Mean_Mean_Intensity', 'Mean_Max_Intensity', 'Mean_Min_Intensity',
            'Mean_Convex_Area', 'Mean_Extent', 'Mean_Eccentricity', 'Mean_Equivalent_Diameter',
            'Classification_Label'
        ]
        writer.writerow(header)


def calculate_regionprops(binary_image, gray_image):
    label_image = label(binary_image)
    regions = regionprops(label_image, intensity_image=gray_image)

    if regions:
        region = regions[0]
        area = region.area
        perimeter = region.perimeter
        mean_intensity = region.mean_intensity
        max_intensity = region.max_intensity
        min_intensity = region.min_intensity
        convex_area = region.convex_area
        extent = region.extent
        eccentricity = region.eccentricity
        equivalent_diameter = region.equivalent_diameter
    else:
        area = perimeter = mean_intensity = max_intensity = min_intensity = 0
        convex_area = extent = eccentricity = equivalent_diameter = 0

    return [
        area, perimeter, mean_intensity, max_intensity, min_intensity,
        convex_area, extent, eccentricity, equivalent_diameter
    ]


def set_camera(camera_id):
    if camera_id == 'A':
        i2c_command = "i2cset -y 1 0x70 0x00 0x04"
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)

    elif camera_id == 'B':
        i2c_command = "i2cset -y 1 0x70 0x00 0x05"
        gp.output(7, True)
        gp.output(11, False)
        gp.output(12, True)

    elif camera_id == 'D':
        i2c_command = "i2cset -y 1 0x70 0x00 0x07"
        gp.output(7, True)
        gp.output(11, True)
        gp.output(12, False)

    os.system(i2c_command)


def capture_and_process(camera_id, cam_num, roi):
    set_camera(camera_id)

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (3280, 2464)},
                                               transform=libcamera.Transform(hflip=1, vflip=1))
    picam2.configure(config)
    picam2.set_controls({
        "AwbEnable": False,  # Turn off auto white balance
        "AeEnable": True,  # Enable auto exposure
        "Brightness": 0.2,
        "AnalogueGain": 7
    })

    picam2.options["quality"] = 95
    picam2.options["compress_level"] = 0
    picam2.start()

    unique_id = datetime.now().strftime("%H%M%S_%Y%m%d")  # Create a unique identifier for the image
    image_filename = os.path.join(RAW_IMAGE_DIR, f"U{unique_id}_C{cam_num}.jpg")
    picam2.capture_file(image_filename)
    picam2.close()

    return process_image(image_filename, cam_num, roi)


def process_image(image_filename, cam_num, roi):
    img = cv2.imread(image_filename)
    y1, y2, x1, x2 = roi
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((4, 4), np.float32) / 25
    median_filter = cv2.filter2D(gray, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_image = clahe.apply(median_filter)
    _, mask = cv2.threshold(cl_image, 75, 225, cv2.THRESH_BINARY + cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    kernel = np.ones((2, 2), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned1 = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    larger_kernel = np.ones((10, 10), np.uint8)
    mask_cleaned_larger = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, larger_kernel)
    mask_cleaned_larger = cv2.morphologyEx(mask_cleaned_larger, cv2.MORPH_OPEN, larger_kernel)

    mask_filled = cv2.morphologyEx(mask_cleaned_larger, cv2.MORPH_CLOSE, larger_kernel)
    mask_dilated = cv2.dilate(mask_filled, larger_kernel, iterations=4)
    mask_eroded = cv2.erode(mask_dilated, larger_kernel, iterations=4)

    mask_filled_improved = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, larger_kernel)
    result_final = cv2.bitwise_and(median_filter, median_filter, mask=mask_filled_improved)
    _, thresh1 = cv2.threshold(result_final, 0, 255, cv2.THRESH_BINARY_INV)  # INVERTED THRESHOLDING

    unique_id = datetime.now().strftime("%H%M%S_%Y%m%d")
    processed_filename = os.path.join(PROCESSED_IMAGE_DIR, f"{unique_id}_C{cam_num}.png")
    cv2.imwrite(processed_filename, thresh1)

    print(f"Processed image saved as {processed_filename}")

    features = calculate_regionprops(thresh1, gray)
    return features


def calculate_mean_features(features_list):
    num_features = len(features_list[0])
    mean_features = []
    for i in range(num_features):
        if isinstance(features_list[0][i], (int, float)):
            mean_features.append(np.mean([f[i] for f in features_list]))
        else:
            mean_features.append(features_list[0][i])
    return mean_features


def start_prediction():
    new_button.config(state="disabled")

    cam1_features = capture_and_process('A', 1, [664, 2309, 492, 3139])
    cam2_features = capture_and_process('B', 2, [648, 2376, 512, 2485])
    cam3_features = capture_and_process('D', 3, [576, 2054, 328, 2859])

    mean_features = calculate_mean_features([cam1_features, cam2_features, cam3_features])
    prediction = model.predict([mean_features])[0]

    # Determine the appropriate image based on prediction
    if prediction == 0:
        result_image_path = "/home/bellpepper/RaspberryPi/GUI/3.png"
    elif prediction == 1:
        result_image_path = "/home/bellpepper/RaspberryPi/GUI/4.png"
    else:
        result_image_path = "/home/bellpepper/RaspberryPi/GUI/5.png"

    result_image = PILImage.open(result_image_path)
    result_image = result_image.resize((1024, 768), PILImage.Resampling.LANCZOS)

    global result_image_photo  # Ensure it stays in scope
    result_image_photo = ImageTk.PhotoImage(result_image)

    # Remove previous prediction image if it exists
    new_canvas.delete("prediction_image")

    new_canvas.create_image(0, 0, image=result_image_photo, anchor="nw", tags="prediction_image")
    new_canvas.image = result_image_photo

    new_button.config(state="normal")
    return prediction


def change_window():
    for widget in window.winfo_children():
        widget.destroy()

    bg_image_path = "/home/bellpepper/RaspberryPi/GUI/2.png"
    new_bg_image = PILImage.open(bg_image_path)
    new_bg_image = new_bg_image.resize((1024, 768), PILImage.Resampling.LANCZOS)

    global new_background_photo  # Ensure it stays in scope
    new_background_photo = ImageTk.PhotoImage(new_bg_image)

    global new_canvas
    new_canvas = Canvas(window, width=1024, height=768)
    new_canvas.pack(fill="both", expand=True)
    new_canvas.create_image(0, 0, image=new_background_photo, anchor="nw")

    global new_button
    new_button = Button(window, text="Start Prediction", fg='white', bg='green', relief=RIDGE,
                        font=("arial", 20, "bold"),
                        padx=5, pady=5, command=start_prediction)
    new_button_window = new_canvas.create_window(button_x, button_y, window=new_button)

    # Create a return button to go back to the initial screen
    return_button = Button(window, text="Return", fg='white', bg='red', relief=RIDGE, font=("arial", 20, "bold"),
                           padx=5, pady=5, command=return_to_initial)
    return_button_window = new_canvas.create_window(button_x, button_y + 60, window=return_button)


def return_to_initial():
    for widget in window.winfo_children():
        widget.destroy()

    # Reinitialize the initial screen
    new_bg_path = "/home/bellpepper/RaspberryPi/GUI/1.png"
    bg_image = PILImage.open(new_bg_path)
    bg_image = bg_image.resize((1024, 768), PILImage.Resampling.LANCZOS)

    global background_photo  # Ensure it stays in scope
    background_photo = ImageTk.PhotoImage(bg_image)

    canvas = Canvas(window, width=1024, height=768)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=background_photo, anchor="nw")

    label1 = Label(window, text="", fg='blue', bg='yellow', relief='solid', font=("arial", 12, "bold"))
    label1.pack()

    button1 = Button(window, text="START", fg='white', bg='green', relief=RIDGE, font=("arial", 20, "bold"),
                     padx=5, pady=5, command=change_window)

    window.update_idletasks()
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    button_width = button1.winfo_reqwidth()
    button_height = button1.winfo_reqheight()
    button_x = (window_width - button_width) // 2
    button_y = window_height - button_height - 180
    button1.place(x=button_x, y=button_y)


# Initialize the main window
window = Tk()
window.title("Prediction GUI")
window.geometry("1024x768")

# Load the initial background image
new_bg_path = "/home/bellpepper/RaspberryPi/GUI/1.png"
bg_image = PILImage.open(new_bg_path)
bg_image = bg_image.resize((1024, 768), PILImage.Resampling.LANCZOS)

global background_photo  # Ensure it stays in scope
background_photo = ImageTk.PhotoImage(bg_image)

# Create a canvas and display the background image
canvas = Canvas(window, width=1024, height=768)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=background_photo, anchor="nw")

# Create a label (empty for now)
label1 = Label(window, text="", fg='blue', bg='yellow', relief='solid', font=("arial", 12, "bold"))
label1.pack()

# Create a button with larger font and padding
button1 = Button(window, text="START", fg='white', bg='green', relief=RIDGE, font=("arial", 20, "bold"),
                 padx=5, pady=5, command=change_window)

window.update_idletasks()
window_width = window.winfo_width()
window_height = window.winfo_height()
button_width = button1.winfo_reqwidth()
button_height = button1.winfo_reqheight()
button_x = (window_width - button_width) // 2
button_y = window_height - button_height - 180
button1.place(x=button_x, y=button_y)
window.mainloop()