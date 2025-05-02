from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox,QComboBox,
                             QStackedWidget,QButtonGroup)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

from clustering import Clustering
from region import RegionGrowing



class Threshold(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thresholding")
        self.setGeometry(200,200,1500,1200)
        self.image = None  # To store the loaded image
        self.processed_image = None
        self.stacked_widget = QStackedWidget()
        self.clustering_page=Clustering(self)
        self.region_page=RegionGrowing(self)


        self.initUI()

    
    def initUI(self):
        
        self.main_widget = QWidget()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.clustering_page)
        self.stacked_widget.addWidget(self.region_page)
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

        # Navigation buttons
        next_pages_buttons_layout = QHBoxLayout()
        self.cluster_button = QPushButton("Clustering")
        self.cluster_button.setFixedWidth(150)
        self.region_button = QPushButton("Region Growing")
        self.region_button.setFixedWidth(150)
        
        next_pages_buttons_layout.addWidget(self.cluster_button)
        next_pages_buttons_layout.addWidget(self.region_button)

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)




        # Noise selection 
        noise_menu_label=QLabel("Threshold menu")
        noise_menu_label.setObjectName("menu")

        threshold_type_label = QLabel("Thresholding Type:")
        self.dropMenu=QComboBox()
        self.dropMenu.addItems(["Otsu Thresholding","Optimal thresholding","spectral thresholding"])


        #Threshold scope selection
        scope_label = QLabel("Thresholding Scope:")
        self.global_radio = QRadioButton("Global")
        self.local_radio = QRadioButton("Local")
        self.global_radio.setChecked(True)


        self.apply_button=QPushButton("Apply Threshold")



        controls_layout.addWidget(noise_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addWidget(threshold_type_label)
        controls_layout.addWidget(self.dropMenu)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(scope_label)
        controls_layout.addWidget(self.global_radio)
        controls_layout.addWidget(self.local_radio)
        controls_layout.addStretch(1)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addStretch(1)


      
        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.apply_button.clicked.connect(self.apply_thresholding)
        self.cluster_button.clicked.connect(self.switch_to_cluster)
        self.region_button.clicked.connect(self.switch_to_region)

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



    def apply_thresholding(self):
        if self.image is None:
            return  # No image loaded
            
        # Convert to grayscale if not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image.copy()
            
        threshold_type = self.dropMenu.currentText()
        is_global = self.global_radio.isChecked()
        
        if threshold_type == "Otsu Thresholding":
            thresholded_image = self.otsu_thresholding(gray_image, is_global)
        elif threshold_type == "Optimal thresholding":
            thresholded_image = self.optimal_thresholding(gray_image, is_global)
        elif threshold_type == "spectral thresholding":
            thresholded_image = self.spectral_thresholding(gray_image, is_global)
            
        self.display_image(thresholded_image, self.output_label)

    def otsu_thresholding(self, image, is_global):
        if is_global:
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()
            
            # Calculate cumulative sum
            cumsum = np.cumsum(hist)
            
            # Calculate cumulative mean
            cummean = np.cumsum(hist * np.arange(256))
            
            # Calculate between-class variance
            between_var = np.zeros(256)
            for t in range(1, 256):
                w0 = cumsum[t]
                w1 = 1 - w0
                if w0 == 0 or w1 == 0:
                    continue
                m0 = cummean[t] / w0
                m1 = (cummean[255] - cummean[t]) / w1
                between_var[t] = w0 * w1 * (m0 - m1) ** 2
            
            # Find optimal threshold
            optimal_threshold = np.argmax(between_var)
            
            thresholded = np.zeros_like(image)
            thresholded[image >= optimal_threshold] = 255
            return thresholded
        else:
            # Local Otsu thresholding
            block_size = 11
            thresholded = np.zeros_like(image)
            for i in range(0, image.shape[0], block_size):
                for j in range(0, image.shape[1], block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    if block.size > 0:
                        local_threshold = self.otsu_thresholding(block, True)
                        thresholded[i:i+block_size, j:j+block_size] = local_threshold
            return thresholded

    def optimal_thresholding(self, image, is_global):
        if is_global:
            # Initialize threshold
            threshold = 128
            prev_threshold = 0
            
            while abs(threshold - prev_threshold) > 0.5:
                # Separate pixels into two classes
                class1 = image[image < threshold]
                class2 = image[image >= threshold]
                
                if len(class1) == 0 or len(class2) == 0:
                    break
                    
                mean1 = np.mean(class1)
                mean2 = np.mean(class2)
                
                # Update threshold
                prev_threshold = threshold
                threshold = (mean1 + mean2) / 2
            
            # Apply threshold
            thresholded = np.zeros_like(image)
            thresholded[image >= threshold] = 255
            return thresholded
        else:
            # Local optimal thresholding
            block_size = 11
            thresholded = np.zeros_like(image)
            for i in range(0, image.shape[0], block_size):
                for j in range(0, image.shape[1], block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    if block.size > 0:
                        local_threshold = self.optimal_thresholding(block, True)
                        thresholded[i:i+block_size, j:j+block_size] = local_threshold
            return thresholded

    def spectral_thresholding(self, image, is_global):
        if is_global:
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()
            
            # Find two thresholds to create three classes
            thresholds = []
            
            # First threshold (separates dark and medium regions)
            between_var = np.zeros(256)
            for t in range(1, 256):
                w0 = np.sum(hist[:t])
                w1 = np.sum(hist[t:])
                if w0 == 0 or w1 == 0:
                    continue
                m0 = np.sum(hist[:t] * np.arange(t)) / w0
                m1 = np.sum(hist[t:] * np.arange(t, 256)) / w1
                between_var[t] = w0 * w1 * (m0 - m1) ** 2
            
            threshold1 = np.argmax(between_var)
            thresholds.append(threshold1)
            
            # Second threshold (separates medium and bright regions)
            between_var = np.zeros(256)
            for t in range(threshold1 + 1, 256):
                w0 = np.sum(hist[threshold1:t])
                w1 = np.sum(hist[t:])
                if w0 == 0 or w1 == 0:
                    continue
                m0 = np.sum(hist[threshold1:t] * np.arange(threshold1, t)) / w0
                m1 = np.sum(hist[t:] * np.arange(t, 256)) / w1
                between_var[t] = w0 * w1 * (m0 - m1) ** 2
            
            threshold2 = np.argmax(between_var)
            thresholds.append(threshold2)
            
            # Sort thresholds
            thresholds = sorted(thresholds)
            
            # Apply thresholds
            thresholded = np.zeros_like(image)
            thresholded[image >= thresholds[0]] = 128
            thresholded[image >= thresholds[1]] = 255
            return thresholded
        else:
            # Local spectral thresholding
            block_size = 11
            thresholded = np.zeros_like(image)
            for i in range(0, image.shape[0], block_size):
                for j in range(0, image.shape[1], block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    if block.size > 0:
                        local_threshold = self.spectral_thresholding(block, True)
                        thresholded[i:i+block_size, j:j+block_size] = local_threshold
            return thresholded

    def switch_to_cluster(self):
        self.stacked_widget.setCurrentIndex(1)

    def switch_to_region(self):
        self.stacked_widget.setCurrentIndex(2)


        

