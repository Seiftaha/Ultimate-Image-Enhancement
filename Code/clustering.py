from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import random

class Clustering(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Clustering")
        self.setGeometry(200, 200, 1500, 1200)
        self.main_window = main_window
        self.image = None
        self.processed_image = None
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
        
        # Color mode selection
        self.color_mode = QRadioButton("Color")
        self.gray_mode = QRadioButton("Grayscale")
        self.color_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
        mode_layout.addWidget(self.gray_mode)
        input_image_layout.addWidget(self.input_label)
        input_image_layout.addLayout(mode_layout)

        # Output image setup
        self.output_label = QLabel("Clustered Image")
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

        # Clustering menu 
        cluster_menu_label = QLabel("Clustering Menu")
        cluster_menu_label.setObjectName("menu")

        # Algorithm selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["K-Means", "Agglomerative", "Mean Shift"])
        
        # Parameters
        self.k_label = QLabel("Number of clusters (K):")
        self.k_input = QLineEdit("3")
        
        self.bw_label = QLabel("Bandwidth (Mean Shift):")
        self.bw_input = QLineEdit("20")

        # Apply button
        self.apply_button = QPushButton("Apply Clustering")
        self.apply_button.clicked.connect(self.apply_clustering)
        
        # Layout for controls
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.method_combo)
        params_layout.addWidget(self.k_label)
        params_layout.addWidget(self.k_input)
        params_layout.addWidget(self.bw_label)
        params_layout.addWidget(self.bw_input)
        
        controls_layout.addWidget(cluster_menu_label)
        controls_layout.addLayout(params_layout)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.apply_button)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.home_button.clicked.connect(self.switch_to_homepage)

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
            QComboBox, QLineEdit {
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
            if self.gray_mode.isChecked():
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if len(self.image.shape) == 2:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            else:
                self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.display_image(self.image, self.input_label)

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

    def save_output_image(self):
        if self.processed_image is None:
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)

    
    def apply_clustering(self):
        if self.image is None:
            return
        
        method = self.method_combo.currentText()
        
        if method == "K-Means":
            k = int(self.k_input.text())
            self.processed_image = self.kmeans_clustering(self.image, k)
        elif method == "Agglomerative":
            k = int(self.k_input.text())
            self.processed_image = self.agglomerative_clustering(self.image, k)
        elif method == "Mean Shift":
            bandwidth = float(self.bw_input.text())
            self.processed_image = self.mean_shift_clustering(self.image, bandwidth)
        
        self.display_image(self.processed_image, self.output_label)

    def kmeans_clustering(self, img, k):
        """K-means clustering implementation"""
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Initialize centroids randomly
        np.random.seed(42)
        centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False), :]
        
        for _ in range(10):  # Fixed number of iterations
            # Assign pixels to nearest centroid
            distances = np.zeros((pixels.shape[0], k))
            for i in range(k):
                distances[:,i] = np.sum((pixels - centroids[i])**2, axis=1)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) > 0:
                    new_centroids[i] = np.mean(cluster_pixels, axis=0)
            
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        
        # Create segmented image
        segmented = np.zeros_like(pixels)
        for i in range(k):
            segmented[labels == i] = centroids[i]
        
        return segmented.reshape(img.shape).astype(np.uint8)

    def agglomerative_clustering(self, img, k):
        """Fast and memory-efficient Agglomerative clustering"""
        # Downsample the image if too large
        if img.shape[0] * img.shape[1] > 2000:  # Further reduced for faster processing
            scale = (2000 / (img.shape[0] * img.shape[1])) ** 0.5
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            
        
        # Convert to float32 for better precision
        pixels = img.reshape((-1, 3)).astype(np.float32)
        n = pixels.shape[0]
        
        # Initialize clusters
        cluster_centers = pixels.copy()
        cluster_sizes = np.ones(n)
        cluster_labels = np.arange(n)
        
        # Use a more efficient distance calculation
        def compute_distances(centers, points):
            return np.sum((centers[:, np.newaxis] - points[np.newaxis, :])**2, axis=2)
        
        while len(np.unique(cluster_labels)) > k:
            # Find valid clusters
            valid_clusters = np.where(cluster_sizes > 0)[0]
            if len(valid_clusters) <= k:
                break
            
            # Compute distances between cluster centers
            distances = compute_distances(cluster_centers[valid_clusters], cluster_centers[valid_clusters])
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            
            # Find closest pair
            min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
            merge_i, merge_j = valid_clusters[min_dist_idx[0]], valid_clusters[min_dist_idx[1]]
            
            # Merge clusters
            total_size = cluster_sizes[merge_i] + cluster_sizes[merge_j]
            cluster_centers[merge_i] = (cluster_centers[merge_i] * cluster_sizes[merge_i] + 
                                      cluster_centers[merge_j] * cluster_sizes[merge_j]) / total_size
            cluster_sizes[merge_i] = total_size
            cluster_sizes[merge_j] = 0
            
            # Update labels
            cluster_labels[cluster_labels == cluster_labels[merge_j]] = cluster_labels[merge_i]
        
        # Create output image
        segmented = np.zeros_like(pixels)
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            mask = cluster_labels == label
            if np.any(mask):
                segmented[mask] = cluster_centers[mask][0]
        
        # Ensure values are in valid range
        segmented = np.clip(segmented, 0, 255)
        
        return segmented.reshape(img.shape).astype(np.uint8)

    def mean_shift_clustering(self, img, bandwidth):
        """Optimized Mean shift clustering implementation"""
        pixels = img.reshape((-1, 3)).astype(np.float32)
        n = pixels.shape[0]
        
        # Use a subset of points as seeds for faster convergence
        seed_indices = np.random.choice(n, min(1000, n), replace=False)
        seeds = pixels[seed_indices]
        
        for i, seed in enumerate(seeds):
            center = seed.copy()
            while True:
                # Find points within bandwidth
                distances = np.linalg.norm(pixels - center, axis=1)
                in_window = pixels[distances < bandwidth]
                
                if len(in_window) == 0:
                    break
                
                # Compute new center
                new_center = np.mean(in_window, axis=0)
                
                # Check for convergence
                if np.linalg.norm(new_center - center) < 0.1 * bandwidth:
                    break
                
                center = new_center
            
            # Assign all points in window to this center
            pixels[distances < bandwidth] = center
        
        return pixels.reshape(img.shape).astype(np.uint8)

    def switch_to_homepage(self):
        self.main_window.stacked_widget.setCurrentIndex(0)
