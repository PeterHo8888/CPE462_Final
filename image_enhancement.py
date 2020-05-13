import cv2
import math
import tkinter as tk
import numpy as np
import skimage as sk

from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import gaussian_filter
from scipy import signal as sk_signal
from scipy import linalg as sk_linalg
from skimage.segmentation import slic as sk_slic
from skimage.exposure import rescale_intensity

from util import create_sliders, register

class ImageEnhancement:
    def __init__(self, window, width, height, canvas_width):
        self.window = window
        self.width = width
        self.height = height
        self.canvas_width = canvas_width

        self.loading_label = None
        self.img_label = None

        self.img = None
        self.pil = None
        self.r = None
        self.g = None
        self.b = None
        register(self.swap_img)

    def loading(self):
        if self.loading_label != None:
            self.loading_label.destroy()
            self.loading_label = None
        self.loading_label = tk.Label(self.window, text = "Loading...", anchor = tk.CENTER, font = ("Helvetica", 24))
        self.loading_label.grid(row = 0, column = 0)
        self.window.update()

    def revert(self):
        self.loading()
        self.r = np.copy(self.origr)
        self.g = np.copy(self.origg)
        self.b = np.copy(self.origb)
        self.swap_img()

    def swap_img(self):
        # Replace channels and generate new pil image
        self.img[...,0] = self.r
        self.img[...,1] = self.g
        self.img[...,2] = self.b

        pil_image = Image.fromarray(self.img)

        width = pil_image.size[0]
        height = pil_image.size[1]

        # I'm using the whole window right now
        if width > self.width or height > self.height:
            if width > height:
                height = int((height / width) * self.width)
                width = self.width
            else:
                width = int((width / height) * self.height)
                height = self.height
            pil_image = pil_image.resize((int(width), int(height)), Image.ANTIALIAS)
    
        if self.img_label != None:
            self.close()

        img_show = ImageTk.PhotoImage(pil_image)
        self.img_label = tk.Label(self.window, image = img_show)
        self.img_label.image = img_show # doesn't work without this
               
        self.img_label.grid(row = 0, column = 0)
        self.img_loaded = 1
        self.window.update()

        
    def open(self):
        self.window.filename = tk.filedialog.askopenfile(initialdir = ".", title = "Select File", filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp *.tif"),))
        string = self.window.filename.name
        print(string)
        
        self.loading()

        self.img = np.array(Image.open(string), dtype = np.uint8)

        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]

        self.origr = np.copy(self.r)
        self.origg = np.copy(self.g)
        self.origb = np.copy(self.b)

        self.swap_img()

    def close(self):
        self.img_label.destroy()
        self.img_label = None
        if self.loading_label != None:
            self.loading_label.destroy()
            self.loading_label = None
        self.window.update()
        

    def save(self):
        self.window.filename = tk.filedialog.asksaveasfile(initialdir=".", title="Save as", filetypes=(("png files", "*.png"),))
        string = self.window.filename.name
        print(string)
        
        Image.fromarray(self.img).save(string, "PNG")

    def closeapp(self):
        self.window.destroy()


    def gradient_y(self):
        self.loading()
        grad_ry = np.gradient(self.r)[0]
        grad_gy = np.gradient(self.g)[0]
        grad_by = np.gradient(self.b)[0]

        self.r = self.normalize2d(grad_ry, 0, 255)
        self.g = self.normalize2d(grad_gy, 0, 255)
        self.b = self.normalize2d(grad_by, 0, 255)

        self.swap_img()

    def gradient_x(self):
        self.loading()
        grad_rx = np.gradient(self.r)[1]
        grad_gx = np.gradient(self.g)[1]
        grad_bx = np.gradient(self.b)[1]

        self.r = self.normalize2d(grad_rx, 0, 255)
        self.g = self.normalize2d(grad_gx, 0, 255)
        self.b = self.normalize2d(grad_bx, 0, 255)

        self.swap_img()

    def rotate(self):
        self.loading()
        height = self.img.shape[0]
        width = self.img.shape[1]
        self.img = np.rot90(self.img, 3)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]
        self.swap_img()

    def flip_vert(self):
        self.loading()
        self.img = np.flipud(self.img)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]
        self.swap_img()

    def flip_hori(self):
        self.loading()
        self.img = np.fliplr(self.img)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]
        self.swap_img()

    def histogram(self):

        colors = ['Red', 'Green', 'Blue']

        hist_r = np.histogram(self.r.ravel(), bins = range(256))[0]
        hist_g = np.histogram(self.g.ravel(), bins = range(256))[0]
        hist_b = np.histogram(self.b.ravel(), bins = range(256))[0]


        hist_r = (hist_r - hist_r.min()) / (np.ptp(hist_r))
        hist_g = (hist_g - hist_g.min()) / (np.ptp(hist_g))
        hist_b = (hist_b - hist_b.min()) / (np.ptp(hist_b))

        fig, ax= plt.subplots()

        x = np.arange(0, 255, 1)
        ax.fill_between(x, 0, hist_r, interpolate = True, facecolor = 'red')
        ax.fill_between(x, 0, hist_g, interpolate = True, facecolor = 'green')
        ax.fill_between(x, 0, hist_b, interpolate = True, facecolor = 'blue')
        
        hist = plt.xlabel('Intensity Value')
        hist = plt.ylabel('Count')
        hist = plt.legend(['Red Channel', 'Green Channel', 'Blue Channel'])

        plt.show()

    def gaussian_blur(self):
        create_sliders(lambda sx, sy: self.fft_convolve2d(self.GaussianFilter, sx, sy), "Gaussian Blur", ("sig_x", 1, 50, 10), ("sig_y", 1, 50, 10))


    def box_blur(self):
        create_sliders(lambda size: self.box_blur_box_blur(size), "Box Blur", ("Kernel size:", 1, 100, 50))

    def box_blur_box_blur(self, size):
        self.loading()
        kernel = np.ones((size, size)) / size ** 2
        self.img = cv2.filter2D(self.img, -1, kernel)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]
        self.swap_img()

    def median_blur(self):
        create_sliders(lambda size: self.median_blur_median_blur(size), "Median Blur", ("Kernel size:", 1, 100, 50))        
    
    def median_blur_median_blur(self, size):
        self.loading()
        self.img = cv2.medianBlur(self.img, 5)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]     
        self.swap_img()

    def sharpen(self):
        self.loading()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.img = cv2.filter2D(self.img, -1, kernel)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]
        self.swap_img()

# Attempt at unsharp masking
#        orig = np.copy(self.img)
#
#        self.fft_convolve2d(self.GaussianFilter, 75, 75)  # larger sigma = less blur
#        blurred = np.zeros(self.img.shape)
#        blurred[...,0] = self.r
#        blurred[...,1] = self.g
#        blurred[...,2] = self.b
#
#        amount = 1
#
#        result = np.zeros(self.img.shape)
#        for c in range(3):
#            result[...,c] = orig[...,c] + (orig[...,c] - blurred[...,c]) * amount
#
#        result = self.normalize2d(result)
#
#        self.r = result[...,0]
#        self.g = result[...,1]
#        self.b = result[...,2]
#
#        self.swap_img()

    def sharpen_kernel(self, kernel):
        ret = np.zeros((self.img.shape[0], self.img.shape[1]))
        cx = self.img.shape[1] // 2
        cy = self.img.shape[0] // 2
        for y in range(-1, 2):
            for x in range(-1, 2):
                ret[cy + y][cx + x] = kernel[y+1][x+1]
                print(ret[cy+y][cx+x])

        return ret

    def reduce_noise(self):
        self.loading()
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 21)
        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]    
        self.swap_img()

    def edge_dectection(self):
        self.loading()
        self.r = cv2.Canny(self.img, 100, 100)
        self.g = cv2.Canny(self.img, 100, 100)
        self.b = cv2.Canny(self.img, 100, 100)
        self.swap_img()

    def harris_detection(self):
        self.loading()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 10, 3, 0.04)
        dst = cv2.dilate(dst, None)

        self.g[dst > 0.01 * dst.max()] = 255

        self.swap_img()

    def hough_transform(self):
        self.loading()
        dst = cv2.Canny(self.img, 50, 200, None, 3)
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        self.r = cdstP[...,0]
        self.g = cdstP[...,1]
        self.b = cdstP[...,2]

        self.swap_img()

    def kmeans(self):
        create_sliders(lambda k: self.kmeans_kmeans(k), "Posterization", ("Levels", 1, 20, 2))
        self.swap_img()

    def kmeans_kmeans(self, K):
        self.loading()
        Z = self.img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((self.img.shape))

        self.r = res2[...,0]
        self.g = res2[...,1]
        self.b = res2[...,2]

        self.swap_img()

    def segment_colorfulness(self, mask):
        B, G, R = cv2.split(self.img.astype("float"))
        R = np.ma.masked_array(R, mask = mask)
        G = np.ma.masked_array(B, mask = mask)
        B = np.ma.masked_array(B, mask = mask)

        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)

        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
        return stdRoot + (0.3 * meanRoot)

    def slic(self):
        self.loading()

        vis = np.zeros(self.img.shape[:2], dtype="float")
        segments = sk_slic(self.img, n_segments = 100, sigma = 5)

        for v in np.unique(segments):
            mask = np.ones(self.img.shape[:2])
            mask[segments == v] = 0

            C = self.segment_colorfulness(mask)
            vis[segments == v] = C

        vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
        alpha = 0.6
        overlay = np.dstack([vis] * 3)
        output = self.img.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        self.r = output[...,0]
        self.g = output[...,1]
        self.b = output[...,2]

        self.swap_img()

    def GaussianFilter(self, sigma_x, sigma_y):
        self.loading()
        nrows = self.img.shape[0]
        ncols = self.img.shape[1]
        cy, cx = nrows / 2, ncols / 2
        x = np.linspace(0, ncols, ncols)
        y = np.linspace(0, nrows, nrows)
        X, Y = np.meshgrid(x, y)
        gmask = np.exp(-(((X - cx) / sigma_x) ** 2 + ((Y - cy) / sigma_y) ** 2))

        return gmask

    def fft_convolve2d(self, kernel_fn, *args):
        fft_img_r = np.fft.fft2(self.img[..., 0])
        fft_img_g = np.fft.fft2(self.img[..., 1])
        fft_img_b = np.fft.fft2(self.img[..., 2])


        fft_img_shift_r = np.fft.fftshift(fft_img_r)
        fft_img_shift_g = np.fft.fftshift(fft_img_g)
        fft_img_shift_b = np.fft.fftshift(fft_img_b)


        kernel = kernel_fn(*args)

        fft_imagep_r = fft_img_shift_r * kernel
        fft_imagep_g = fft_img_shift_g * kernel
        fft_imagep_b = fft_img_shift_b * kernel

        fft_ishift_r = np.fft.ifftshift(fft_imagep_r)
        fft_ishift_g = np.fft.ifftshift(fft_imagep_g)
        fft_ishift_b = np.fft.ifftshift(fft_imagep_b)


        imagep_r = np.fft.ifft2(fft_ishift_r)
        imagep_g = np.fft.ifft2(fft_ishift_g)
        imagep_b = np.fft.ifft2(fft_ishift_b)

        self.r = np.abs(imagep_r)
        self.g = np.abs(imagep_g)
        self.b = np.abs(imagep_b)

        self.r = self.normalize2d(self.r)
        self.g = self.normalize2d(self.g)
        self.b = self.normalize2d(self.b)

    def normalize2d(self, img, min_ = 0, max_ = 255):
        return (img - img.min()) / (np.ptp(img)) * max_ + min_

    def human_pose_estimation(self):
        self.loading()

        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

        inWidth = 368
        inHeight = 368
        threshold = 0.05

        net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

        frame = np.copy(self.img)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            points.append((int(x), int(y)) if conf > threshold else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(self.img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(self.img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(self.img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        self.r = self.img[...,0]
        self.g = self.img[...,1]
        self.b = self.img[...,2]

        self.swap_img()
