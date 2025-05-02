from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt

from filters import Filter
from edge_detectors import EdgeDetector
from threshold import Threshold
from frequency_filters import FrequencyFilters
from hybrid import Hybrid
from equalize import Equalizer
from hist_equalize import HistogramEqualizer



class Noiser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal viewer")
        self.setGeometry(200,200,1500,1200)
        self.image = None  # To store the loaded image
        self.noisy_image = None  # To store the loaded image

        self.stacked_widget=QStackedWidget()
        self.filters_page = Filter(self)
        self.edge_page=EdgeDetector(self)
        self.threshold_page=Threshold(self)
        self.frequency_filters_page=FrequencyFilters(self)
        self.hybrid_page=Hybrid(self)
        self.equalizer_page=Equalizer(self)
        self.hist_page=HistogramEqualizer(self)

        self.initUI()
    
    def initUI(self):
        
        self.main_widget = QWidget()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.filters_page)
        self.stacked_widget.addWidget(self.edge_page)
        self.stacked_widget.addWidget(self.threshold_page)
        self.stacked_widget.addWidget(self.frequency_filters_page)
        self.stacked_widget.addWidget(self.hybrid_page)
        self.stacked_widget.addWidget(self.equalizer_page)
        self.stacked_widget.addWidget(self.hist_page)
        self.stacked_widget.setCurrentWidget(self.main_widget)


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

        self.output_label = QLabel("Noisy Image")
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
        self.filters_button=QPushButton("Filters")
        self.filters_button.setFixedWidth(150)
        self.edges_button=QPushButton("Edge Detectors")
        self.edges_button.setFixedWidth(150)
        self.thresholds_button=QPushButton("Thresholds")
        self.thresholds_button.setFixedWidth(150)
        self.frequency_filters_button=QPushButton("Frequency filters")
        self.frequency_filters_button.setFixedWidth(150)
        self.hybrid_page_button=QPushButton("Hybrid images")
        self.hybrid_page_button.setFixedWidth(150)
        self.equalizer_page_button=QPushButton("Image enhancement")
        self.equalizer_page_button.setFixedWidth(180)
        self.hist_page_button=QPushButton("Histogram Equalizer")
        self.hist_page_button.setFixedWidth(180)
        next_pages_buttons_layout.addWidget(self.filters_button)
        next_pages_buttons_layout.addWidget(self.edges_button)
        next_pages_buttons_layout.addWidget(self.thresholds_button)
        next_pages_buttons_layout.addWidget(self.frequency_filters_button)
        next_pages_buttons_layout.addWidget(self.hybrid_page_button)
        next_pages_buttons_layout.addWidget(self.equalizer_page_button)
        next_pages_buttons_layout.addWidget(self.hist_page_button)

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)




        # Noise selection 
        noise_menu_label=QLabel("Noise menu")
        noise_menu_label.setObjectName("menu")

        #  Gaussian noise
        mean_layout=QHBoxLayout()
        mean_layout.addWidget(QLabel("Mean: "))
        self.mean_input=QLineEdit()
        mean_layout.addWidget(self.mean_input)
        deviation_layout=QHBoxLayout()
        deviation_layout.addWidget(QLabel("Standard deviation: "))
        self.deviation_input=QLineEdit()
        deviation_layout.addWidget(self.deviation_input)
        self.apply_gaussian=QPushButton("Apply Gaussian")

        # salt & pepper noise
        noise_ratio_layout=QHBoxLayout()
        noise_ratio_layout.addWidget(QLabel("Noise amount"))
        self.slider_salt=QSlider(Qt.Horizontal)
        self.slider_salt.setMinimum(1)
        self.slider_salt.setMaximum(100)
        self.slider_salt.setValue(10)
        noise_ratio_layout.addWidget(self.slider_salt)
        self.apply_salt=QPushButton("Apply salt and pepper")

        # Uniform noise 
        uniform_slider_layout=QHBoxLayout()
        uniform_slider_layout.addWidget(QLabel("Noise amount"))
        self.slider_uniform=QSlider(Qt.Horizontal)
        self.slider_uniform.setMinimum(1)
        self.slider_uniform.setMaximum(100)
        self.slider_uniform.setValue(10)
        uniform_slider_layout.addWidget(self.slider_uniform)
        self.apply_uniform=QPushButton("Apply uniform")

        self.histogram_CDF_button=QPushButton("Histogram and CDF")

        controls_layout.addWidget(noise_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Gaussian noise: "))
        controls_layout.addLayout(mean_layout)
        controls_layout.addLayout(deviation_layout)
        controls_layout.addWidget(self.apply_gaussian)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Salt & Pepper noise: "))
        controls_layout.addLayout(noise_ratio_layout)
        controls_layout.addWidget(self.apply_salt)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Uniform noise: "))
        controls_layout.addLayout(uniform_slider_layout)
        controls_layout.addWidget(self.apply_uniform)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)

      
        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.apply_gaussian.clicked.connect(lambda: self.apply_noise("Gaussian"))
        self.apply_uniform.clicked.connect(lambda: self.apply_noise("Uniform"))
        self.apply_salt.clicked.connect(lambda: self.apply_noise("Salt & Pepper"))
        self.filters_button.clicked.connect(self.switch_to_filters)
        self.edges_button.clicked.connect(self.switch_to_edges)
        self.thresholds_button.clicked.connect(self.switch_to_thresholds)
        self.frequency_filters_button.clicked.connect(self.switch_to_freq_filters)
        self.hybrid_page_button.clicked.connect(self.switch_to_hybrid)
        self.histogram_CDF_button.clicked.connect(self.show_histogram_CDF)
        self.equalizer_page_button.clicked.connect(self.switch_to_equalizer)
        self.hist_page_button.clicked.connect(self.switch_to_hist_equalizer)

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
        
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.stacked_widget)

            
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


    
    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    
    def apply_noise(self, type):
        if self.image is None:
            return
        
        noise_type = type
        self.noisy_image = self.image.copy()
        
        if noise_type == "Gaussian":
            mean = float(self.mean_input.text()) 
            std_dev = float(self.deviation_input.text()) 
            gauss = np.random.normal(mean, std_dev, self.image.shape).astype(np.uint8)
            self.noisy_image = cv2.add(self.image, gauss)
        elif noise_type == "Uniform":
            intensity=self.slider_uniform.value()/100
            uniform_noise = np.random.uniform(-intensity * 255, intensity * 255, self.image.shape).astype(np.uint8)
            self.noisy_image = cv2.add(self.image, uniform_noise)
        elif noise_type == "Salt & Pepper":
            value=self.slider_salt.value()/100
            prob = value / 10  # Scale probability
            self.noisy_image = self.image.copy()

            # Check if grayscale or color
            if len(self.image.shape) == 2:  # Grayscale image (2D)
                h, w = self.image.shape
                num_salt = np.ceil(prob * h * w * 0.5)
                num_pepper = np.ceil(prob * h * w * 0.5)

                # Add salt (white pixels)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.image.shape]
                self.noisy_image[coords[0], coords[1]] = 255

                # Add pepper (black pixels)
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.image.shape]
                self.noisy_image[coords[0], coords[1]] = 0

            else:  # Color image (3D)
                h, w, c = self.image.shape
                num_salt = np.ceil(prob * h * w * 0.5)
                num_pepper = np.ceil(prob * h * w * 0.5)

                # Add salt (white pixels)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.image.shape[:2]]
                self.noisy_image[coords[0], coords[1], :] = 255

                # Add pepper (black pixels)
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.image.shape[:2]]
                self.noisy_image[coords[0], coords[1], :] = 0
         
        
        self.display_image(self.noisy_image, self.output_label)

    def show_histogram_CDF(self):
        if self.image is None or self.noisy_image is None:
            return  # Ensure both images are loaded

        # Check if grayscale or color
        is_grayscale = len(self.image.shape) == 2

        plt.figure(figsize=(10, 8))

        # Function to compute histogram and CDF
        def compute_histogram_cdf(image):
            if is_grayscale:
                hist, _ = np.histogram(image.flatten(), 256, [0, 256])
                cdf = hist.cumsum() * hist.max() / hist.cumsum().max()  
                return [hist], [cdf]
            else:
                hist = [np.histogram(image[:, :, i].flatten(), 256, [0, 256])[0] for i in range(3)]
                cdf = [h.cumsum() * h.max() / h.cumsum().max() for h in hist] 
                return hist, cdf

        hist_input, cdf_input = compute_histogram_cdf(self.image)
        hist_noisy, cdf_noisy = compute_histogram_cdf(self.noisy_image)

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
            plt.plot(cdf_noisy[i], color=color, linestyle='dashed', linewidth=2, label=f"processed CDF - {color.capitalize()}")
            plt.bar(range(256), hist_noisy[i], color=color, alpha=0.4, label=f"processed Histogram - {color.capitalize()}")
        plt.title("processed Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()


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



    def switch_to_filters(self):
        self.stacked_widget.setCurrentIndex(1)  # Show Filters Page
    def switch_to_edges(self):
        self.stacked_widget.setCurrentIndex(2)  # Show Filters Page

    def switch_to_thresholds(self):
        self.stacked_widget.setCurrentIndex(3)

    def switch_to_freq_filters(self):
        self.stacked_widget.setCurrentIndex(4)
    
    def switch_to_hybrid(self):
        self.stacked_widget.setCurrentIndex(5)

    def switch_to_equalizer(self):
        self.stacked_widget.setCurrentIndex(6)
    def switch_to_hist_equalizer(self):
        self.stacked_widget.setCurrentIndex(7)



