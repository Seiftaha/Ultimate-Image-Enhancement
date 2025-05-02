from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class FrequencyFilters(QMainWindow):
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
        self.home_page_button=QPushButton("Home page")
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




        # Noise selection 
        noise_menu_label=QLabel("Frequency filters menu")
        noise_menu_label.setObjectName("menu")

        self.filters_combobox=QComboBox()
        self.filters_combobox.addItems(["Low pass filter","High pass filter"])

        self.radius_slider=QSlider(Qt.Horizontal)

        self.apply_button=QPushButton("Apply filter")

        self.histogram_CDF_button=QPushButton("Histogram and CDF")




        controls_layout.addWidget(noise_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Filter type:"))
        controls_layout.addWidget(self.filters_combobox)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Radius:"))
        controls_layout.addWidget(self.radius_slider)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)

      
        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.home_page_button.clicked.connect(self.switch_to_home_page)
        self.apply_button.clicked.connect(self.apply_selected_filter)
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
        """Loads an image and computes its frequency domain representation."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )

        if file_path:
            # Check which mode is selected
            if self.gray_mode.isChecked():
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                self.display_image(self.image, self.input_label)  # Show original image

            else:
                self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load as color (default)
                self.display_image(self.image, self.input_label)  # Show original image
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for FFT


            # Compute Frequency Domain (FFT)
            self.fft_image = np.fft.fft2(self.image)
            self.fft_shift = np.fft.fftshift(self.fft_image)  # Shift low frequencies to center

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

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
        plt.title("processed Image Histogram & CDF")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()


    def create_filter(self, shape, radius, highpass=False):
        """Creates a frequency domain filter."""
        rows, cols = shape
        center_x, center_y = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)

        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                if highpass:
                    mask[i, j] = 1 if dist > radius else 0  # High-Pass Filter
                else:
                    mask[i, j] = 1 if dist < radius else 0  # Low-Pass Filter

        return mask
    
    def apply_selected_filter(self):
        """Applies the selected frequency domain filter and displays the output."""
        if self.fft_shift is None:
            return  # No image uploaded

        radius = self.radius_slider.value()  # Get radius from slider
        selected_filter = self.filters_combobox.currentText()  # Get filter type

        # Determine High-Pass or Low-Pass
        highpass = True if selected_filter == "High pass filter" else False
        filter_mask = self.create_filter(self.image.shape, radius, highpass)

        # Apply the mask in the frequency domain
        filtered_shift = self.fft_shift * filter_mask
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_shift)).real  # Inverse FFT

        # 🔥 Convert to 8-bit (0-255)
        filtered_image = np.clip(filtered_image, 0, 255)  # Ensure values are within range
        self.filtered_image = np.uint8(filtered_image)  # Convert to uint8

        self.display_image(self.filtered_image, self.output_label)  # Show filtered output

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

    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)