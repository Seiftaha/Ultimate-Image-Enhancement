from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt





class EdgeDetector(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        self.setWindowTitle("Signal viewer")
        self.setGeometry(200,200,1500,1200)
        self.main_window=main_window
        self.image = None  # To store the loaded image
        self.equalized_image = None  # To store the loaded image


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

        self.output_label = QLabel("Equalized Image")
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




        # Mask selection 
        noise_menu_label=QLabel("Edge Detectors menu")
        noise_menu_label.setObjectName("menu")

        #  Sobel mask
        self.apply_sobel_button=QPushButton("Apply Sobel mask")

        # Prewitt mask
        self.apply_prewitt_button=QPushButton("Apply Prewitt mask")

        # Robert mask
        self.apply_robert_button=QPushButton("Apply Robert mask")

        # Canny mask
        high_layout=QHBoxLayout()
        high_layout.addWidget(QLabel("High threshold: "))
        self.high_input=QLineEdit()
        high_layout.addWidget(self.high_input)
        low_layout=QHBoxLayout()
        low_layout.addWidget(QLabel("Low threshold: "))
        self.low_input=QLineEdit()
        low_layout.addWidget(self.low_input)
        self.apply_canny_button=QPushButton("Apply Canny mask")

        self.histogram_CDF_button=QPushButton("Histogram and CDF")






        controls_layout.addWidget(noise_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Sobel mask: "))
        controls_layout.addWidget(self.apply_sobel_button)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Prewitt mask: "))
        controls_layout.addWidget(self.apply_prewitt_button)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Robert mask: "))
        controls_layout.addWidget(self.apply_robert_button)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel("Canny mask: "))
        controls_layout.addLayout(high_layout)
        controls_layout.addLayout(low_layout)
        controls_layout.addWidget(self.apply_canny_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_CDF_button)
        controls_layout.addStretch(1)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.apply_canny_button.clicked.connect(self.apply_canny)
        self.apply_sobel_button.clicked.connect(self.apply_sobel)
        self.apply_prewitt_button.clicked.connect(self.apply_prewitt)
        self.apply_robert_button.clicked.connect(self.apply_robert)
        self.histogram_CDF_button.clicked.connect(self.show_histogram_CDF)


        main_layout.addLayout(controls_layout,0,0)
        main_layout.addWidget(group_box,0,1)
        main_layout.setColumnStretch(1,2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)


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

    # def show_histogram_CDF(self):
    #     if self.image is None:
    #         return  # No image loaded

    #     # Check if the image is grayscale or colored
    #     if len(self.image.shape) == 2:  # Grayscale image
    #         hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
    #         cdf = hist.cumsum()
    #         cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF to match histogram scale

    #         plt.figure(figsize=(8, 6))
    #         plt.plot(cdf_normalized, color='black', linestyle='dashed', linewidth=2, label="CDF")
    #         plt.bar(range(256), hist, color='gray', alpha=0.6, label="Histogram")
    #         plt.title("Grayscale Histogram & CDF")
    #         plt.xlabel("Pixel Intensity")
    #         plt.ylabel("Frequency")
    #         plt.legend()

    #     else:  # Colored image (BGR)
    #         colors = ('blue', 'green', 'red')  # OpenCV loads images in BGR format
    #         plt.figure(figsize=(8, 6))

    #         for i, color in enumerate(colors):
    #             hist, bins = np.histogram(self.image[:, :, i].flatten(), 256, [0, 256])
    #             cdf = hist.cumsum()
    #             cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF

    #             plt.plot(cdf_normalized, color=color, linestyle='dashed', linewidth=2, label=f"CDF - {color.capitalize()}")
    #             plt.bar(range(256), hist, color=color, alpha=0.4, label=f"Histogram - {color.capitalize()}")

    #         plt.title("Color Histogram & CDF")
    #         plt.xlabel("Pixel Intensity")
    #         plt.ylabel("Frequency")
    #         plt.legend()

    #     plt.show()  # Open in a new window

    def show_histogram_CDF(self):
        if self.image is None or self.equalized_image is None:
            return  # Ensure both images are loaded

        # Determine if the input image and filtered image are grayscale or color
        is_input_grayscale = len(self.image.shape) == 2
        is_filtered_grayscale = len(self.equalized_image.shape) == 2

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
        hist_filtered, cdf_filtered = compute_histogram_cdf(self.equalized_image)

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



    def apply_canny(self):
        if self.image is None:
            return

        # Check if the image is colored (3 channels)
        if len(self.image.shape) == 3:  # Color image (BGR)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            gray = self.image.copy()

        # Get user input values
        try:
            low_threshold = int(self.low_input.text())
            high_threshold = int(self.high_input.text())
        except ValueError:
            print("Invalid input! Please enter numeric values.")
            return

        # Apply Canny Edge Detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        self.equalized_image=edges

        # Display the result
        self.display_image(edges, self.output_label)

    def apply_sobel(self):
        """Applies the Sobel edge detection filter using manual convolution."""
        if self.image is not None:
            # Define Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Convert to grayscale if needed
            if len(self.image.shape) == 3:  # If it's a colored image
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image.copy()

            # Apply convolution with roberts=False explicitly
            grad_x = self.convolve(gray_image, sobel_x)
            grad_y = self.convolve(gray_image, sobel_y)

            # Compute gradient magnitude
            sobel_output = np.sqrt(grad_x**2 + grad_y**2)
            sobel_output = np.clip(sobel_output, 0, 255).astype(np.uint8)  # Ensure valid pixel range

            self.equalized_image=sobel_output

            # Display the result
            self.display_image(sobel_output, self.output_label)

    def convolve(self,image, kernel, roberts=False):
        kernel_height, kernel_width = kernel.shape
        
        # Set padding size based on filter type
        if roberts:
            pad_h, pad_w = 1, 1
        else:
            pad_h, pad_w = kernel_height // 2, kernel_width // 2

        # Pad the image to handle border effects
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        output = np.zeros_like(image, dtype=np.float32)

        # Perform convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_height, j:j+kernel_width]
                output[i, j] = np.sum(region * kernel)
        
        return output

    
    def apply_prewitt(self):
        if self.image is None:
            return

        # Convert to grayscale if the image is colored
        gray_image = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Define Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        # Apply convolution
        grad_x = self.convolve(gray_image, prewitt_x)
        grad_y = self.convolve(gray_image, prewitt_y)

        # Compute gradient magnitude
        prewitt_edges = np.sqrt(grad_x**2 + grad_y**2)
        prewitt_edges=np.abs(prewitt_edges)
        prewitt_edges = (prewitt_edges / prewitt_edges.max()) * 255  # Normalize to [0,255]

        self.equalized_image=prewitt_edges

        # Display the result
        self.display_image(prewitt_edges.astype(np.uint8), self.output_label)

    def apply_robert(self):
        if self.image is None:
            return

        # Convert to grayscale if needed
        gray_image = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Roberts kernels
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # Apply convolution
        grad_x = self.convolve(gray_image, roberts_x,roberts=True)
        grad_y = self.convolve(gray_image, roberts_y,roberts=True)

        # Compute gradient magnitude
        roberts_edges = np.sqrt(grad_x**2 + grad_y**2)
        roberts_edges = np.abs(roberts_edges)  # Take absolute values

        # Normalize properly
        roberts_edges = (roberts_edges / np.max(roberts_edges)) * 255 if np.max(roberts_edges) != 0 else roberts_edges

        self.equalized_image=roberts_edges

        # Display the result
        self.display_image(roberts_edges.astype(np.uint8), self.output_label)



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


