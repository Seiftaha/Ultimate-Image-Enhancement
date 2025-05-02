from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Hybrid(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        self.setWindowTitle("Signal viewer")
        self.setGeometry(200,200,1500,1200)
        self.main_window=main_window
        self.image1 = None  # To store the loaded image
        self.image2 = None
        self.filtered_image = None  # To store the loaded image
        self.kernel_size = 3  # Default kernel size

        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QGridLayout()
        

        group_box = QGroupBox()
        box_layout=QVBoxLayout()
        images_layout=QHBoxLayout()
        buttons_layout=QHBoxLayout()

        # Labels for images
        input_image1_layout=QVBoxLayout()
        self.input1_label = QLabel("First Image")
        self.input1_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input1_label.setAlignment(Qt.AlignCenter)
        self.input1_label.setFixedSize(500, 400)
        self.color1_mode = QRadioButton("Color")
        self.gray1_mode = QRadioButton("Grayscale")
        self.color1_mode.setChecked(True)  # Default mode is Color
        mode1_layout = QHBoxLayout()
        mode1_layout.addWidget(self.color1_mode)
        mode1_layout.addWidget(self.gray1_mode)
        mode1_layout.addStretch(1)
        input_image1_layout.addWidget(self.input1_label)
        input_image1_layout.addLayout(mode1_layout)


        input_image2_layout=QVBoxLayout()
        self.input2_label = QLabel("Second Image")
        self.input2_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input2_label.setAlignment(Qt.AlignCenter)
        self.input2_label.setFixedSize(500, 400)
        self.color2_mode = QRadioButton("Color")
        self.gray2_mode = QRadioButton("Grayscale")
        self.color2_mode.setChecked(True)  # Default mode is Color
        mode2_layout = QHBoxLayout()
        mode2_layout.addWidget(self.color2_mode)
        mode2_layout.addWidget(self.gray2_mode)
        mode2_layout.addStretch(1)
        input_image2_layout.addWidget(self.input2_label)
        input_image2_layout.addLayout(mode2_layout)



        self.output_label = QLabel("Hybrid Image")
        self.output_label.setStyleSheet("color:gray; background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 400)

        images_layout.addStretch(1)
        images_layout.addLayout(input_image1_layout)
        images_layout.addStretch(1)
        images_layout.addLayout(input_image2_layout)
        images_layout.addStretch(1)

        self.mix_button = QPushButton("Apply mix")
        self.mix_button.setFixedWidth(150)
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFixedWidth(150)
        self.reset_button=QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button=QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.mix_button)
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

        output_label_layout=QHBoxLayout()
        output_label_layout.addStretch(1)
        output_label_layout.addStretch(1)
        output_label_layout.addWidget(self.output_label)
        output_label_layout.addStretch(1)
        output_label_layout.addStretch(1)

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(output_label_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.mix_button.clicked.connect(self.apply_hybrid)

        
        main_layout.addWidget(group_box,0,0)
        

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if self.image1 is None:
                self.image1 = img
                self.display_image(self.image1, self.input1_label)
            else:
                self.image2 = img
                self.display_image(self.image2, self.input2_label)
    

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)


    def reset_images(self):
        self.input1_label.clear()  # Clear input image label
        self.input2_label.clear()
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


    def low_pass_filter(self, image, kernel_size=15, sigma=5):
        h, w = image.shape[:2]
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        
        # Create Gaussian kernel manually
        for x in range(-kernel_size//2, kernel_size//2 + 1):
            for y in range(-kernel_size//2, kernel_size//2 + 1):
                kernel[x + kernel_size//2, y + kernel_size//2] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        kernel /= np.sum(kernel)  # Normalize

        # Apply filter manually (convolution)
        filtered = np.zeros_like(image, dtype=np.float32)
        pad = kernel_size // 2
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        for i in range(h):
            for j in range(w):
                filtered[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel[:, :, None], axis=(0, 1))

        return np.clip(filtered, 0, 255).astype(np.uint8)
    

    def high_pass_filter(self, image, kernel_size=15, sigma=5):
        low_pass = self.low_pass_filter(image, kernel_size, sigma)
        high_pass = image.astype(np.float32) - low_pass.astype(np.float32)
        return np.clip(high_pass + 128, 0, 255).astype(np.uint8)  # Shift to make visible
    

    def apply_hybrid(self):
        if self.image1 is None or self.image2 is None:
            return  # Ensure both images are loaded
        
        # Resize image2 to match image1
        self.image2 = cv2.resize(self.image2, (self.image1.shape[1], self.image1.shape[0]))

        low_pass_img = self.low_pass_filter(self.image1)
        high_pass_img = self.high_pass_filter(self.image2)

        hybrid_img = np.clip(low_pass_img.astype(np.float32) + high_pass_img.astype(np.float32) - 128, 0, 255).astype(np.uint8)

        self.display_image(hybrid_img, self.output_label)
        self.hybrid_image = hybrid_img


    def switch_to_home_page(self):
        self.main_window.stacked_widget.setCurrentIndex(0)


