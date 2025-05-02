from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HistogramEqualizer(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        self.setWindowTitle("Signal viewer")
        self.setGeometry(200,200,1500,1200)
        self.main_window=main_window
        self.image = None # To store the loaded image
        self.equalized_image = None  # To store the loaded image
        self.grayscale_image=None
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
        self.color_mode.setChecked(True)  # Default mode is Color
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
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



        # Controls selection 
        Enhancement_menu_label=QLabel("Controls menu")
        Enhancement_menu_label.setObjectName("menu")


        # convert to grayscale
        convert_layout = QHBoxLayout()

        self.apply_conversion= QPushButton("Apply")
        self.apply_conversion.clicked.connect(self.convert_to_grayscale)
        

        

        #  Equalize 
        equalizer_layout=QHBoxLayout()
        
        self.apply_equalize= QPushButton("Apply")
        self.apply_equalize.clicked.connect(self.apply_equalize_function)
        self.histogram_CDF_button = QPushButton("Histogram and CDF")
        self.histogram_CDF_button.clicked.connect(self.show_histogram_CDF)
                     
        controls_layout.addWidget(Enhancement_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Convert to Grayscale"))
        controls_layout.addLayout(convert_layout)
        controls_layout.addWidget(self.apply_conversion)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Equalize"))
        controls_layout.addLayout(equalizer_layout)
        controls_layout.addWidget(self.apply_equalize)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)

        



        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)    
        

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
            if self.color_mode.isChecked():
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


    def convert_to_grayscale(self):
        if self.image is None:
            return  # No image loaded
        
        # Convert the BGR color image to a grayscale image using OpenCV function
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Convert the image to QImage format
        height, width = self.grayscale_image.shape
        bytes_per_line = width
        q_image = QImage(self.grayscale_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Display the image
        pixmap = QPixmap.fromImage(q_image).scaled(self.output_label.width(), self.output_label.height(), Qt.KeepAspectRatio)
        self.output_label.setPixmap(pixmap)




    def equalize(self, image):
        if len(image.shape) == 2:  # Grayscale image
            return self.equalize_channel(image)
        else:  # Color image
            channels = cv2.split(image)
            equalized_channels = [self.equalize_channel(channel) for channel in channels]
            return cv2.merge(equalized_channels)

    def equalize_channel(self, channel):
        # Compute the histogram
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])

        # Compute the cumulative distribution function (CDF)
        cdf = hist.cumsum()

        # Normalize the CDF
        cdf_normalized = 255 * cdf / cdf[-1]
        cdf_normalized = cdf_normalized.astype(np.uint8)

        # Use the normalized CDF to map the original pixel values to equalized values
        equalized_channel = cdf_normalized[channel]

        return equalized_channel


    

    def apply_equalize_function(self):
        if self.grayscale_image is None:
            return  # No grayscale image loaded

        # Apply custom equalize function to the grayscale image
        self.equalized_image = self.equalize(self.grayscale_image)

        # Convert the image to QImage format
        height, width = self.equalized_image.shape
        bytes_per_line = width
        q_image = QImage(self.equalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Display the image
        pixmap = QPixmap.fromImage(q_image).scaled(self.output_label.width(), self.output_label.height(), Qt.KeepAspectRatio)
        self.output_label.setPixmap(pixmap)



    

    def display_image(self, img, label):
        if len(img.shape) == 3:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = img.shape
            bytes_per_line = width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    

    def show_histogram_CDF(self):
        if self.image is None or (self.grayscale_image is None and self.equalized_image is None):
            return  # Ensure both images are loaded

        plt.figure(figsize=(10, 8))

        # Function to compute histogram and CDF
        def compute_histogram_cdf(image):
            hist, _ = np.histogram(image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF
            return hist, cdf_normalized

        # Compute histograms & CDFs for input image
        if len(self.image.shape) == 3:  # Color image
            hist_input = [np.histogram(self.image[:, :, i].flatten(), 256, [0, 256])[0] for i in range(3)]
            cdf_input = [h.cumsum() * h.max() / h.cumsum().max() for h in hist_input]
        else:  # Grayscale image
            hist_input, cdf_input = compute_histogram_cdf(self.image)

        # Determine which processed image to use (grayscale or equalized)
        if self.equalized_image is not None:
            processed_image = self.equalized_image
        else:
            processed_image = self.grayscale_image

        hist_processed, cdf_processed = compute_histogram_cdf(processed_image)

        colors = ('blue', 'green', 'red') if len(self.image.shape) == 3 else ['black']

        # Plot input image histogram & CDF
        plt.subplot(2, 1, 1)
        if len(self.image.shape) == 3:
            for i, color in enumerate(colors):
                plt.plot(cdf_input[i], color=color, linestyle='dashed', linewidth=2, label=f"Input CDF - {color.capitalize()}")
                plt.bar(range(256), hist_input[i], color=color, alpha=0.4, label=f"Input Histogram - {color.capitalize()}")
        else:
            plt.plot(cdf_input, color='black', linestyle='dashed', linewidth=2, label="Input CDF")
            plt.bar(range(256), hist_input, color='gray', alpha=0.4, label="Input Histogram")
        plt.title("Input Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot processed image histogram & CDF
        plt.subplot(2, 1, 2)
        if len(processed_image.shape) == 3:
            for i, color in enumerate(colors):
                plt.plot(cdf_processed[i], color=color, linestyle='dashed', linewidth=2, label=f"Processed CDF - {color.capitalize()}")
                plt.bar(range(256), hist_processed[i], color=color, alpha=0.4, label=f"Processed Histogram - {color.capitalize()}")
        else:
            plt.plot(cdf_processed, color='black', linestyle='dashed', linewidth=2, label="Processed CDF")
            plt.bar(range(256), hist_processed, color='gray', alpha=0.4, label="Processed Histogram")
        plt.title("Processed Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()


    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)
