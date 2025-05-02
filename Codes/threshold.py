from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Threshold(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        self.setWindowTitle("Signal viewer")
        self.setGeometry(200,200,1500,1200)
        self.main_window=main_window
        self.image = None # To store the loaded image
        self.filtered_image = None  # To store the loaded image
        self.kernel_size = 3  # Default kernel size

        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()

        group_box = QGroupBox()
        box_layout=QVBoxLayout()
        images_layout=QHBoxLayout()
        buttons_layout=QHBoxLayout()

        # Labels for images
        input_image_layout=QVBoxLayout()
        self.input_label = QLabel("Original Image")
        self.input_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedSize(500, 500)
        self.color_mode = QRadioButton("Color")
        self.gray_mode = QRadioButton("Grayscale")
        self.color_mode.setChecked(True)  # Default mode is Color
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
        mode_layout.addWidget(self.gray_mode)
        input_image_layout.addWidget(self.input_label)
        input_image_layout.addLayout(mode_layout)

        self.output_label = QLabel("Filtered Image")
        self.output_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 500)


        images_layout.addLayout(input_image_layout)
        images_layout.addWidget(self.output_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFixedWidth(150)
        self.reset_button=QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button=QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # Next pages buttons
        next_pages_buttons_layout=QHBoxLayout()
        self.home_page_button=QPushButton("Home page")
        self.home_page_button.clicked.connect(self.switch_to_home_page)
        self.home_page_button.setFixedWidth(150)
        next_pages_buttons_layout.addWidget(self.home_page_button)
        next_pages_buttons_layout.addStretch(1)

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)



        # Thresholds selection 
        Thresholds_menu_label=QLabel("Thresholds menu")
        Thresholds_menu_label.setObjectName("menu")

        #  Global thresholding
        global_thresholding_layout=QHBoxLayout()
        self.slider_global= QSlider(Qt.Horizontal)
        self.slider_global.setMinimum(0)
        self.slider_global.setMaximum(255)
        self.slider_global.setValue(128)
        # self.slider_global.valueChanged.connect(self.update_image)  # Connect slider to update_image
        global_thresholding_layout.addWidget(self.slider_global)
        self.apply_global= QPushButton("Apply Global Thresholding")
        self.apply_global.clicked.connect(self.update_image)
                     
        controls_layout.addWidget(Thresholds_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Global thresholding: "))
        controls_layout.addLayout(global_thresholding_layout)
        controls_layout.addWidget(self.apply_global)
        controls_layout.addStretch(1)
     
        # controls_layout.addStretch(1)
        # controls_layout.addStretch(1)

        

        # Mean adaptive thresholding
        local_thresholding_layout = QHBoxLayout()
        self.slider_local = QSlider(Qt.Horizontal)
        self.slider_local.setMinimum(-10)
        self.slider_local.setMaximum(10)
        self.slider_local.setValue(2)
        # self.slider_local.valueChanged.connect(self.apply_mean_adaptive_thresholding)  # Connect slider to apply_mean_adaptive_thresholding
        local_thresholding_layout.addWidget(self.slider_local)
        self.apply_local = QPushButton("Apply Local Thresholding")
        self.histogram_CDF_button=QPushButton("Histogram and CDF")


        controls_layout.addWidget(Thresholds_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Local Thresholding: "))
        controls_layout.addLayout(local_thresholding_layout)
        controls_layout.addWidget(self.apply_local)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)
        



        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)    
        self.apply_local.clicked.connect(self.apply_mean_adaptive_thresholding)
        self.histogram_CDF_button.clicked.connect(self.show_histogram_CDF)
        

        main_layout.addLayout(controls_layout,0,0)
        main_layout.addWidget(group_box,0,1)
        main_layout.setColumnStretch(1,2)

        self.setStyleSheet("""
             QLabel{
                font-size:20px;
                color:white;     
                    }
            QLabel#menu{
                font-size:29px;
                color:white;
                           }
            QPushButton{
                    font-size:18px;
                    padding:10px;
                    border:white 1px solid;
                    border-radius:15px;
                    background-color:white;
                    color:black;         
                        }

        """)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )

        if file_path:
            # Check which mode is selected
            if self.gray_mode.isChecked():
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            else:
                self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load as color (default)

            self.display_image(self.image, self.input_label)  # Display in input label

    def reset_images(self):
        self.input_label.clear()  # Clear input image label
        self.output_label.clear()  # Clear output image label
        self.image = None  # Remove stored image
        self.noisy_image = None  # Remove stored output

    def save_output_image(self):
        if self.noisy_image is None:
            return  # No image to save
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
        )
        
        if file_path:
            cv2.imwrite(file_path, self.noisy_image)  # Save using OpenCV

    def show_histogram_CDF(self):
        if self.image is None or self.filtered_image is None:
            return  # Ensure both images are loaded

        # Check if grayscale or color
        is_grayscale = len(self.image.shape) == 2

        plt.figure(figsize=(10, 8))

        # Function to compute histogram and CDF
        def compute_histogram_cdf(image):
            if is_grayscale:
                hist, _ = np.histogram(image.flatten(), 256, [0, 256])
                cdf = hist.cumsum() * hist.max() / hist.cumsum().max()  # Normalize CDF
                return [hist], [cdf]
            else:
                hist = [np.histogram(image[:, :, i].flatten(), 256, [0, 256])[0] for i in range(3)]
                cdf = [h.cumsum() * h.max() / h.cumsum().max() for h in hist]  # Normalize CDF
                return hist, cdf

        # Compute histograms & CDFs for input image
        hist_input, cdf_input = compute_histogram_cdf(self.image)

        # Determine which processed image to use (equalized or normalized)
        hist_input, cdf_input = compute_histogram_cdf(self.image)
        hist_filtered, cdf_filtered = compute_histogram_cdf(self.filtered_image)

        colors = ('blue', 'green', 'red') if not is_grayscale else ['black']

        # Plot input image histogram & CDF
        plt.subplot(2, 1, 1)
        for i, color in enumerate(colors):
            plt.plot(cdf_input[i], color=color, linestyle='dashed', linewidth=2, label=f"Input CDF - {color.capitalize()}")
            plt.bar(range(256), hist_input[i], color=color, alpha=0.4, label=f"Input Histogram - {color.capitalize()}")
        plt.title("Input Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot processed image histogram & CDF
        plt.subplot(2, 1, 2)
        for i, color in enumerate(colors):
            plt.plot(cdf_filtered[i], color=color, linestyle='dashed', linewidth=2, label=f"processed CDF - {color.capitalize()}")
            plt.bar(range(256), hist_filtered[i], color=color, alpha=0.4, label=f"processed Histogram - {color.capitalize()}")
        plt.title("Processed Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def global_threshold(self, image, threshold_value):
        # If the image is colored, convert it to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        # Create a copy of the original image
        thresholded_image = np.zeros_like(image)
        
        # Apply the threshold
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] < threshold_value:
                    thresholded_image[i, j] = 0
                else:
                    thresholded_image[i, j] = 255
        
        return thresholded_image

    def update_image(self):
        if self.image is None:
            return  # No image loaded

        # Get the current slider value
        threshold_value = self.slider_global.value()
        
        # Apply custom global threshold
        thresholded_image = self.global_threshold(self.image, threshold_value)

        self.filtered_image= thresholded_image

        # Convert the image to QImage format
        height, width = thresholded_image.shape
        bytesPerLine = width
        q_image = QImage(thresholded_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        
        # Display the image
        pixmap = QPixmap.fromImage(q_image)
        self.output_label.setPixmap(pixmap)
        self.output_label.resize(pixmap.width(), pixmap.height())


    

    def mean_adaptive_threshold(self, image, block_size, C):
        # If the image is colored, convert it to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a copy of the original image
        thresholded = np.zeros_like(image)
        
        # Define the half block size
        half_block = block_size // 2
        
        # Iterate through the image
        for i in range(half_block, image.shape[0] - half_block):
            for j in range(half_block, image.shape[1] - half_block):
                # Calculate the local mean
                local_mean = np.mean(image[i - half_block:i + half_block + 1, j - half_block:j + half_block + 1])
                
                # Apply the threshold
                if image[i, j] < local_mean - C:
                    thresholded[i, j] = 0
                else:
                    thresholded[i, j] = 255
        
        return thresholded

    def apply_mean_adaptive_thresholding(self):
        if self.image is None:
            return  # No image loaded
        
        block_size = 11  # Default block size
        C = self.slider_local.value()  # Get value from slider

        # Apply custom mean adaptive threshold
        thresholded_image = self.mean_adaptive_threshold(self.image, block_size, C)

        # Update the filtered image
        self.filtered_image = thresholded_image

        # Convert the image to QImage format
        height, width = thresholded_image.shape
        bytesPerLine = width
        q_image = QImage(thresholded_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        
        # Display the image
        pixmap = QPixmap.fromImage(q_image)
        self.output_label.setPixmap(pixmap)
        self.output_label.resize(pixmap.width(), pixmap.height())


    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    

    # def display_image(self, image, label):
    #     height, width = image.shape[:2]
    #     bytesPerLine = 3 * width if len(image.shape) == 3 else width
    #     q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888 if len(image.shape) == 3 else QImage.Format_Grayscale8)
    #     pixmap = QPixmap.fromImage(q_image)
    #     label.setPixmap(pixmap)
    #     label.resize(pixmap.width(), pixmap.height())



    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)