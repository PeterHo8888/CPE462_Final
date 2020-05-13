import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import ImageTk, Image
from image_enhancement import ImageEnhancement

if __name__ == '__main__':

    window = tk.Tk()
    window.resizable(False, False)

    width = 1920 // 2
    height = 1080 // 2

    canvas_width = width

    canvas_sides = (width - canvas_width) / 2

    IE = ImageEnhancement(window, width, height, canvas_width)
    window.title("Image Converter App")
    icon = tk.PhotoImage(file = './images/app_logo.png')
    window.geometry("%dx%d+0+0" % (width, height))

    window.grid_columnconfigure(0, weight = 1)

    menu = tk.Menu(window)
    window.config(menu = menu)

# File Menu
    file_menu = tk.Menu(menu, tearoff=0)
    menu.add_cascade(label = "File", menu=file_menu)
    file_menu.add_command(label ="Open", command = IE.open)
    file_menu.add_command(label = "Save", command = IE.save)
    file_menu.add_command(label = "Revert", command = IE.revert)
    file_menu.add_command(label = "Close Image", command = IE.close)
    file_menu.add_command(label = "Exit App", command = IE.closeapp)

# Transform
    transform_menu = tk.Menu(menu, tearoff = 0)
    menu.add_cascade(label = "Transform", menu = transform_menu)
    transform_menu.add_command(label = "Rotate", command = IE.rotate)
    transform_menu.add_command(label = "Flip vertical", command = IE.flip_vert)
    transform_menu.add_command(label = "Flip horizontal", command = IE.flip_hori)

# Filter
    filter_menu = tk.Menu(menu, tearoff = 0)
    blur_menu = tk.Menu(file_menu, tearoff = 0)
    menu.add_cascade(label="Filter", menu=filter_menu)
    filter_menu.add_command(label = "Gradient Y", command = IE.gradient_y)
    filter_menu.add_command(label = "Gradient X", command = IE.gradient_x)
    filter_menu.add_cascade(label = "Blurs", menu = blur_menu)
    blur_menu.add_command(label = "Gaussian Blur", command = IE.gaussian_blur)
    blur_menu.add_command(label = "Box Blur", command = IE.box_blur)
    filter_menu.add_command(label = "Sharpening", command = IE.sharpen)
    filter_menu.add_command(label = "Reduce Noise", command = IE.reduce_noise)

# Data
    data_menu = tk.Menu(menu, tearoff = 0)
    menu.add_cascade(label = "Data", menu = data_menu)
    data_menu.add_command(label = "Histogram", command = IE.histogram)

# Detection
    detection_menu = tk.Menu(menu, tearoff = 0)
    menu.add_cascade(label = "Detections", menu = detection_menu)
    detection_menu.add_command(label = "Edge Dectection", command = IE.edge_dectection)
    detection_menu.add_command(label = "Corner Dectection", command = IE.harris_detection)
    detection_menu.add_command(label = "Line Dectection", command = IE.hough_transform)

# Segmentation
    segementation_menu = tk.Menu(menu, tearoff = 0)
    menu.add_cascade(label = "Segmentation", menu = segementation_menu)
    segementation_menu.add_command(label = "K-Means", command = IE.kmeans)
    segementation_menu.add_command(label = "SLIC", command = IE.slic)

# Machine learning
    machine_learning_menu = tk.Menu(menu, tearoff = 0)
    menu.add_cascade(label = "Machine Learning", menu =  machine_learning_menu)
    machine_learning_menu.add_command(label = "Human Pose Estimation", command = IE.human_pose_estimation)

    canvas = tk.Canvas(window, width = width, height = height)
    canvas.grid(row = 0, column = 0)

    window.mainloop()
