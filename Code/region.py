from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
                            QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

class RegionGrowing(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Region Growing Segmentation")
        self.setGeometry(200, 200, 1500, 1200)
        self.main_window = main_window
        self.image = None
        self.processed_image = None
        self.seed_points = []
        self.initUI()

    def initUI(self):  
        self.main_widget = QWidget()
        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()

        group_box = QGroupBox()
        box_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()

        # Input image setup
        input_image_layout = QVBoxLayout()
        self.input_label = QLabel("Original Image")
        self.input_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedSize(500, 500)
        input_image_layout.addWidget(self.input_label)

        # Output image setup
        self.output_label = QLabel("Segmented Image")
        self.output_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 500)

        images_layout.addLayout(input_image_layout)
        images_layout.addWidget(self.output_label)

        # Action buttons
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFixedWidth(150)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button = QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # Navigation button
        self.home_button = QPushButton("Home")
        self.home_button.setFixedWidth(150)

        box_layout.addWidget(self.home_button)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)

        # Segmentation parameters
        seg_menu_label = QLabel("Region Growing Parameters")
        seg_menu_label.setObjectName("menu")

        self.threshold_label = QLabel("Similarity Threshold:")
        self.threshold_input = QLineEdit("20")
        self.clear_seeds_button = QPushButton("Clear Seeds")
        self.clear_seeds_button.clicked.connect(self.clear_seeds)
        self.apply_button = QPushButton("Apply Segmentation")
        self.apply_button.clicked.connect(self.apply_segmentation)
        
        # Layout for controls
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.threshold_label)
        params_layout.addWidget(self.threshold_input)
        params_layout.addWidget(self.clear_seeds_button)
        
        controls_layout.addWidget(seg_menu_label)
        controls_layout.addLayout(params_layout)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.apply_button)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.home_button.clicked.connect(self.switch_to_homepage)

        # Connect mouse click event
        self.input_label.mousePressEvent = self.get_seed_point

        main_layout.addLayout(controls_layout, 0, 0)
        main_layout.addWidget(group_box, 0, 1)
        main_layout.setColumnStretch(1, 2)

        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

        self.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: white;
            }
            QLabel#menu {
                font-size: 29px;
                color: white;
            }
            QPushButton {
                font-size: 18px;
                padding: 10px;
                border: white 1px solid;
                border-radius: 15px;
                background-color: white;
                color: black;
            }
            QLineEdit {
                font-size: 16px;
                padding: 5px;
                border: 1px solid gray;
                border-radius: 5px;
                min-width: 150px;
            }
            QGroupBox {
                border: 1px solid gray;
                margin-top: 10px;
            }
        """)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options)
        
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.display_image(self.image, self.input_label)
            self.seed_points = []  # Clear previous seeds when loading new image

    def display_image(self, img, label):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, ch = img.shape
        bytes_per_line = 3 * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def reset_images(self):
        self.input_label.clear()
        self.output_label.clear()
        self.image = None
        self.processed_image = None
        self.seed_points = []

    def save_output_image(self):
        if self.processed_image is None:
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)

    def apply_segmentation(self):
        if self.image is None:
            print("Error: No image loaded")
            return
        
        if len(self.seed_points) == 0:
            print("Error: No seed points selected")
            return
        
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            print("Error: Invalid threshold value")
            return
        
        # Convert seed points from (x,y) to (row,col) format
        seeds = [(y, x) for (x, y) in self.seed_points]
        
        # Convert image to grayscale
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply region growing
        segmentation_mask = self.region_grow_segmentation(gray_img, seeds, threshold)
        
        # Create color output (green for segmented regions)
        self.processed_image = self.image.copy()
        self.processed_image[segmentation_mask == 1] = [0, 255, 0]
        
        # Add boundary visualization
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.processed_image, contours, -1, (0, 0, 255), 2)
        
        # Display the result
        self.display_image(self.processed_image, self.output_label)
        print("Segmentation completed successfully")

    def get_seed_point(self, event):
        """Handles mouse clicks to select seed points"""
        if self.image is not None and event.button() == Qt.LeftButton:
            # Get click position relative to label
            x = event.pos().x()
            y = event.pos().y()

            # Check if click is inside label bounds
            if 0 <= x <= self.input_label.width() and 0 <= y <= self.input_label.height():
                # Convert to image coordinates
                img_x = int(x * self.image.shape[1] / self.input_label.width())
                img_y = int(y * self.image.shape[0] / self.input_label.height())
                
                if (0 <= img_x < self.image.shape[1]) and (0 <= img_y < self.image.shape[0]):
                    self.seed_points.append((img_x, img_y))
                    self.show_seed_points()
                    print(f"Seed point added at: ({img_x}, {img_y})")  # Optional, for debugging

    def show_seed_points(self):
        """Displays seed points on the input image"""
        if self.image is None:
            return
            
        display_img = self.image.copy()
        for (x, y) in self.seed_points:
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)  # Red circles for seeds
        self.display_image(display_img, self.input_label)

    def clear_seeds(self):
        """Clears all selected seed points"""
        self.seed_points = []
        if self.image is not None:
            self.display_image(self.image, self.input_label)

    def get_difference(self, img, current_point, point_2):
        """Calculate the difference between two pixels considering their neighborhood"""
        x1, y1 = current_point
        x2, y2 = point_2
        
        # Get 3x3 neighborhood around each point
        def get_neighborhood(x, y):
            return img[max(0, x-1):min(img.shape[0], x+2), 
                      max(0, y-1):min(img.shape[1], y+2)]
        
        # Calculate mean intensity of neighborhoods
        n1 = get_neighborhood(x1, y1)
        n2 = get_neighborhood(x2, y2)
        
        # Return the maximum difference between the neighborhoods
        return abs(int(n1.mean()) - int(n2.mean()))

    def get_around_pixels(self):
        around = [(1, -1), (1, 0), (1, 1), 
                 (0, -1),          (0, 1),
                 (-1, -1), (-1, 0), (-1, 1)]
        return around

    def region_grow_segmentation(self, img, seeds, threshold):
        """Improved region growing segmentation algorithm"""
        height, width = img.shape
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)
        neighbors = self.get_around_pixels()
        
        # Process each seed point separately
        for seed in seeds:
            current_y, current_x = seed
            
            # Skip if already processed
            if segmentation_mask[current_y, current_x] == 1:
                continue
                
            # Initialize queue for this region
            queue = [(current_y, current_x)]
            region_mean = float(img[current_y, current_x])
            region_size = 1
            
            while queue:
                current_point = queue.pop(0)
                current_y, current_x = current_point
                
                # Skip if already processed
                if segmentation_mask[current_y, current_x] == 1:
                    continue
                    
                segmentation_mask[current_y, current_x] = 1
                
                for dy, dx in neighbors:
                    neighbor_y = current_y + dy
                    neighbor_x = current_x + dx
                    
                    # Check bounds
                    if (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                        # Check if pixel meets threshold and not already processed
                        if segmentation_mask[neighbor_y, neighbor_x] == 0:
                            # Calculate difference with current region mean
                            diff = abs(int(img[neighbor_y, neighbor_x]) - region_mean)
                            
                            if diff < threshold:
                                # Add to queue and update region statistics
                                queue.append((neighbor_y, neighbor_x))
                                # Update region mean incrementally
                                region_mean = (region_mean * region_size + img[neighbor_y, neighbor_x]) / (region_size + 1)
                                region_size += 1
        
        return segmentation_mask

    def switch_to_homepage(self):
        self.main_window.stacked_widget.setCurrentIndex(0)