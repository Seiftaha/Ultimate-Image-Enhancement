from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Filter(QMainWindow):
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



        # Filters selection 
        filters_menu_label=QLabel("Filters menu")
        filters_menu_label.setObjectName("menu")

        #  Gaussian filter
        Sigma_layout=QHBoxLayout()
        Sigma_layout.addWidget(QLabel("Sigma: "))
        self.Sigma_input=QLineEdit()
        Sigma_layout.addWidget(self.Sigma_input)             
        self.apply_gaussian=QPushButton("Apply Gaussian")

        #  kernel size
        kernel_size_layout=QHBoxLayout()
        kernel_size_label=QLabel("Kernel Size: ")
        self.kernel_combo_box = QComboBox()
        self.kernel_combo_box.addItems(["3x3", "5x5", "7x7"])
        kernel_size_layout.addWidget(kernel_size_label)
        kernel_size_layout.addWidget(self.kernel_combo_box)

        # Average filter
        
        self.apply_average=QPushButton("Apply Average")

        # Median Filter 
        self.apply_median=QPushButton("Apply Median")

        self.histogram_CDF_button=QPushButton("Histogram and CDF")




        controls_layout.addWidget(filters_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addLayout(kernel_size_layout)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Gaussian filter: "))
        controls_layout.addLayout(Sigma_layout)
        controls_layout.addWidget(self.apply_gaussian)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Average filter: "))
        controls_layout.addWidget(self.apply_average)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Median filter: "))
        controls_layout.addWidget(self.apply_median)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.apply_gaussian.clicked.connect(self.apply_gaussian_filter)
        self.apply_average.clicked.connect(self.apply_average_filter)
        self.apply_median.clicked.connect(self.apply_median_filter)
        self.kernel_combo_box.currentIndexChanged.connect(lambda: self.set_kernel_size())
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

    def convert_to_grayscale(self):
        """Ensures the image is grayscale before applying filters."""
        if len(self.image.shape) == 3:  # If the image is colored (3 channels)
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        return self.image  # If already grayscale, return as is

    
    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

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

        # Determine if the input image and filtered image are grayscale or color
        is_input_grayscale = len(self.image.shape) == 2
        is_filtered_grayscale = len(self.filtered_image.shape) == 2

        plt.figure(figsize=(10, 8))

        # Function to compute histogram and CDF
        def compute_histogram_cdf(image):
            if len(image.shape) == 2:  # Grayscale image
                hist, _ = np.histogram(image.flatten(), 256, [0, 256])
                cdf = hist.cumsum() * hist.max() / hist.cumsum().max()  # Normalize CDF
                return [hist], [cdf]  # Return as lists for consistent indexing
            else:  # Color image
                hist = [np.histogram(image[:, :, i].flatten(), 256, [0, 256])[0] for i in range(3)]
                cdf = [h.cumsum() * h.max() / h.cumsum().max() for h in hist]  # Normalize CDF
                return hist, cdf

        # Compute histograms & CDFs for input and filtered images
        hist_input, cdf_input = compute_histogram_cdf(self.image)
        hist_filtered, cdf_filtered = compute_histogram_cdf(self.filtered_image)

        # Default colors (will be adjusted below)
        colors = ['black']  # Default to grayscale
        
        # Adjust colors based on image type
        if not is_input_grayscale and not is_filtered_grayscale:  # Both are color
            colors = ['blue', 'green', 'red']
        elif not is_input_grayscale and is_filtered_grayscale:  # Input is color, filtered is grayscale
            colors_input = ['blue', 'green', 'red']
            colors_filtered = ['black']
        elif is_input_grayscale and not is_filtered_grayscale:  # Input is grayscale, filtered is color
            colors_input = ['black']
            colors_filtered = ['blue', 'green', 'red']
        else:  # Both are grayscale
            colors_input = colors_filtered = ['black']

        # Plot input image histogram & CDF
        plt.subplot(2, 1, 1)
        for i, color in enumerate(colors_input):
            plt.plot(cdf_input[i], color=color, linestyle='dashed', linewidth=2, label=f"Input CDF - {color.capitalize()}")
            plt.bar(range(256), hist_input[i], color=color, alpha=0.4, label=f"Input Histogram - {color.capitalize()}")
        plt.title("Input Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot filtered image histogram & CDF
        plt.subplot(2, 1, 2)
        for i, color in enumerate(colors_filtered):
            plt.plot(cdf_filtered[i], color=color, linestyle='dashed', linewidth=2, label=f"Filtered CDF - {color.capitalize()}")
            plt.bar(range(256), hist_filtered[i], color=color, alpha=0.4, label=f"Filtered Histogram - {color.capitalize()}")
        plt.title("Filtered Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()





    def apply_average_filter(self):
        """Applies a manually implemented Average (Mean) filter to a grayscale image."""
        if self.image is not None:
            gray_image = self.convert_to_grayscale()  # Convert if needed
            self.filtered_image = self.convolve(gray_image, np.ones((self.kernel_size, self.kernel_size)) / (self.kernel_size ** 2))
            self.display_image(self.filtered_image, self.output_label)

    def apply_gaussian_filter(self):
        """Applies a manually implemented Gaussian filter to a grayscale image."""
        if self.image is not None:
            gray_image = self.convert_to_grayscale()  # Convert if needed

            try:
                sigma = float(self.Sigma_input.text())  # Get user-defined sigma
            except ValueError:
                sigma = 1.0  # Default sigma if input is invalid

            if self.kernel_size % 2 == 0:  # Ensure kernel size is odd
                self.kernel_size += 1

            kernel = self.gaussian_kernel(self.kernel_size, sigma)  # Generate Gaussian kernel
            self.filtered_image = self.convolve(gray_image, kernel)

            self.display_image(self.filtered_image, self.output_label)



    def apply_median_filter(self):
        """Applies a manually implemented Median filter to a grayscale image."""
        if self.image is not None:
            gray_image = self.convert_to_grayscale()  # Convert if needed
            self.filtered_image = self.median_filter(gray_image, self.kernel_size)
            self.display_image(self.filtered_image, self.output_label)


    def convolve(self, image, kernel):
        """Applies convolution between the image and the given kernel."""
        h, w = image.shape
        k = kernel.shape[0] // 2
        output = np.zeros_like(image, dtype=np.float32)

        for i in range(k, h - k):
            for j in range(k, w - k):
                region = image[i - k:i + k + 1, j - k:j + k + 1]
                output[i, j] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)
    
    def gaussian_kernel(self, size, sigma):
        """Generates a Gaussian kernel manually."""
        k = size // 2  # Get half the kernel size
        x, y = np.mgrid[-k:k+1, -k:k+1]  # Create a coordinate grid

        # Apply the Gaussian formula
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

        return gaussian / gaussian.sum()  # Normalize so that all weights sum to 1

    
    def median_filter(self, image, kernel_size):
        """Applies a manually implemented Median filter."""
        h, w = image.shape
        k = kernel_size // 2
        output = np.zeros_like(image)

        for i in range(k, h - k):
            for j in range(k, w - k):
                region = image[i - k:i + k + 1, j - k:j + k + 1]
                output[i, j] = np.median(region)

        return output.astype(np.uint8)
    
    def set_kernel_size(self):
        """Sets the kernel size for filtering."""
        index = self.kernel_combo_box.currentIndex()
        if index == 0:
            self.kernel_size = 3
        if index == 1:
            self.kernel_size = 5
        if index == 2:
            self.kernel_size = 7



    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)
