"""
User Interface (View) module for the RaptorVision application.
Handles all PySide6 graphical components, layouts, and widget styling.
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFrame, QLabel, QStackedWidget, 
                             QGridLayout, QScrollArea, QSizePolicy, QStatusBar, 
                             QSlider, QSplitter, QGroupBox, QPlainTextEdit, QProgressBar,
                             QCheckBox, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt

class PatchCard(QGroupBox):
    """
    Interactive card component used in the Library Gallery.
    Displays a cached heatmap preview and provides metadata and deletion options.
    
    Args:
        heatmap_pixmap (QPixmap): The pre-rendered heatmap image to display.
        info_text (str): The metadata string (e.g., index and filename).
        index (int): The internal library index of this patch.
        delete_callback (callable): The function to trigger when the delete button is clicked.
    """
    def __init__(self, heatmap_pixmap, info_text, index, delete_callback):
        super().__init__()
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px;")
        layout = QVBoxLayout(self)
        
        # --- Header Section: Info Label & Delete Action ---
        header = QHBoxLayout()
        lbl_info = QLabel(info_text)
        lbl_info.setStyleSheet("font-size: 10px; font-weight: bold; color: #aaa; border: none;")
        
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet("background-color: #721c24; color: white; border: none;")
        btn_del.clicked.connect(lambda: delete_callback(index))
        
        header.addWidget(lbl_info)
        header.addStretch()
        header.addWidget(btn_del)
        
        # --- Image Display Section ---
        lbl_img = QLabel()
        lbl_img.setPixmap(heatmap_pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("background: #000; border: 1px solid #111;")
        
        layout.addLayout(header)
        layout.addWidget(lbl_img)

class PatchImageWidget(QFrame):
    """
    Reusable container widget for displaying neural network images.
    Features a dark-themed title bar and an auto-scaling image area.
    
    Args:
        title (str): The text to display in the header bar.
    """
    def __init__(self, title):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Header Title ---
        self.label_title = QLabel(title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 6px;")
        
        # --- Image Canvas ---
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(1, 1)
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_image.setStyleSheet("background-color: #121212;")
        
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_image, 1)

class MainWindow(QMainWindow):
    """
    Primary Application Window for RaptorVision.
    Implements a multi-view architecture using a QStackedWidget to seamlessly
    switch between Explorer, Library, About, Help, and Processing interfaces.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaptorVision")
        self.resize(1450, 900)
        
        # Initialize top menu bar
        self._create_menubar()

        # Core container holding all application views
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # ==========================================
        # 1. EXPLORER VIEW (Index 0)
        # Main workspace for selecting patches and previewing matches.
        # ==========================================
        self.explorer_view = QWidget()
        exp_layout = QVBoxLayout(self.explorer_view)
        
        # Splitter to allow dynamic resizing of workspace panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.view_local = PatchImageWidget("LOCAL VIEW")
        self.view_memory = PatchImageWidget("MEMORY VIEW")
        
        # --- Right Sidebar: Inspector Panel ---
        self.inspector_panel = QFrame()
        self.inspector_panel.setStyleSheet("background-color: #1e1e1e; border-left: 1px solid #333;")
        ins_layout = QVBoxLayout(self.inspector_panel)
        
        # Heatmap (Match) Preview Section
        ins_layout.addWidget(QLabel("🎯 HEATMAP (MATCH)"), alignment=Qt.AlignCenter)
        self.label_source_preview = QLabel("Select a point")
        self.label_source_preview.setFixedSize(256, 256)
        self.label_source_preview.setStyleSheet("border: 1px solid #444; background: #000;")
        self.label_source_preview.setAlignment(Qt.AlignCenter)
        ins_layout.addWidget(self.label_source_preview)
        
        # Raw Source Image Section
        ins_layout.addWidget(QLabel("🖼️ SOURCE IMAGE"), alignment=Qt.AlignCenter)
        self.label_source_clean = QLabel("Original")
        self.label_source_clean.setFixedSize(256, 256)
        self.label_source_clean.setStyleSheet("border: 1px solid #444; background: #000;")
        self.label_source_clean.setAlignment(Qt.AlignCenter)
        ins_layout.addWidget(self.label_source_clean)

        self.label_source_info = QLabel("")
        self.label_source_info.setStyleSheet("color: #999; font-size: 11px;")
        
        self.btn_delete_patch = QPushButton("🗑️ Delete Patch")
        self.btn_delete_patch.setVisible(False)
        self.btn_delete_patch.setStyleSheet("background-color: #500; color: white; padding: 10px; font-weight: bold;")

        ins_layout.addWidget(self.label_source_info)
        ins_layout.addWidget(self.btn_delete_patch)
        ins_layout.addStretch()

        # Build Splitter Layout
        self.main_splitter.addWidget(self.view_local)
        self.main_splitter.addWidget(self.view_memory)
        self.main_splitter.addWidget(self.inspector_panel)

        # Splitter Layout Ratios: Local(1), Memory(1), Inspector(0/Fixed)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        # --- Bottom Navigation Bar ---
        nav_bar = QFrame()
        nav_bar.setFixedHeight(60)
        nav_bar.setStyleSheet("background: #252525; border-top: 1px solid #333;")
        nav_layout = QHBoxLayout(nav_bar)
        
        self.btn_prev = QPushButton("⬅️ Previous")
        self.btn_cancel = QPushButton("🔄 Cancel")
        self.btn_next = QPushButton("Next ➡️")
        self.btn_next.setStyleSheet("background-color: #2d5a27; font-weight: bold; padding: 8px 25px;")
        
        # Similarity Threshold Slider
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(60)
        self.slider_threshold.setFixedWidth(160)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_cancel)
        nav_layout.addStretch()
        nav_layout.addWidget(QLabel("Threshold:"))
        nav_layout.addWidget(self.slider_threshold)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        
        exp_layout.addWidget(self.main_splitter, 1)
        exp_layout.addWidget(nav_bar)

        # ==========================================
        # 2. EDITION VIEW (Index 1)
        # Grid gallery for managing the active patch library.
        # ==========================================
        self.edit_view = QWidget()
        edi_layout = QVBoxLayout(self.edit_view)
        
        # Library Management Actions
        act_layout = QHBoxLayout()
        self.btn_merge = QPushButton("🔗 Merge Libraries")
        self.btn_delete_lib = QPushButton("🗑️ Delete Library")
        self.btn_merge.setStyleSheet("background-color: #155724; color: white; padding: 12px;")
        self.btn_delete_lib.setStyleSheet("background-color: #721c24; color: white; padding: 12px;")
        act_layout.addWidget(self.btn_merge)
        act_layout.addWidget(self.btn_delete_lib)
        
        # Scrollable Grid Canvas
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.patch_grid = QGridLayout(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        
        edi_layout.addLayout(act_layout)
        edi_layout.addWidget(self.scroll_area)

        # Register views to central stack
        self.central_stack.addWidget(self.explorer_view)
        self.central_stack.addWidget(self.edit_view)

        # ==========================================
        # GLOBAL STATUS BAR
        # Displays active configuration and library metrics.
        # ==========================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.label_lib_name = QLabel("📂 Lib: -")
        self.label_mem_count = QLabel("📦 0")
        self.label_thresh_val = QLabel("🎯 0.60")
        for w in [self.label_lib_name, self.label_mem_count, self.label_thresh_val]:
            w.setStyleSheet("margin-right: 20px; font-weight: bold; color: #ddd;")
            self.status_bar.addPermanentWidget(w)

        # ==========================================
        # 3. ABOUT VIEW (Index 2)
        # Displays software configuration and author information.
        # ==========================================
        self.about_view = QWidget()
        about_layout = QVBoxLayout(self.about_view)
        about_layout.setAlignment(Qt.AlignCenter)

        info_box = QFrame()
        info_box.setFixedWidth(600)
        info_box.setStyleSheet("background-color: #252525; border-radius: 10px; border: 1px solid #444; padding: 20px;")
        info_inner_layout = QVBoxLayout(info_box)

        # Technical Configuration Panel
        sw_header = QLabel("SOFTWARE CONFIGURATION:")
        sw_header.setStyleSheet("font-weight: bold; color: #58a6ff; font-size: 14px; border: none; padding-bottom: 0px;")
        self.lbl_about_software = QLabel("Model: -\nResolution: -\nHardware: -")
        self.lbl_about_software.setStyleSheet("color: #ddd; font-size: 14px; border: none;")

        # Contact & Biography Panel
        author_header = QLabel("CONTACT INFORMATION:")
        author_header.setStyleSheet("font-weight: bold; color: #58a6ff; font-size: 14px; border: none;")
        self.lbl_author_name = QLabel("Name: [Name]")
        self.lbl_author_email = QLabel("Contact: [Email]")
        self.lbl_author_name.setStyleSheet("color: #ddd; font-size: 14px; border: none; padding: 0px; padding-left: 20px;")
        self.lbl_author_email.setStyleSheet("color: #ddd; font-size: 14px; border: none; padding: 0px; padding-left: 20px;")

        self.lbl_author_bio_title = QLabel("About:")
        self.lbl_author_bio_title.setStyleSheet("font-weight: bold; color: #58a6ff; margin-top: 10px; border: none;")
        self.lbl_author_bio_content = QLabel()
        self.lbl_author_bio_content.setWordWrap(True)
        self.lbl_author_bio_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_author_bio_content.setStyleSheet("""
            color: #bbb; 
            background-color: #1a1a1a; 
            border: 1px solid #333; 
            padding: 15px;
            margin-left: 20px;
            margin-right: 20px;
            border-radius: 5px;
            line-height: 150%;
        """)

        # Assemble About View
        info_inner_layout.addWidget(sw_header)
        info_inner_layout.addWidget(self.lbl_about_software)
        info_inner_layout.addWidget(author_header)
        info_inner_layout.addWidget(self.lbl_author_name)
        info_inner_layout.addWidget(self.lbl_author_email)
        info_inner_layout.addWidget(self.lbl_author_bio_title)
        info_inner_layout.addWidget(self.lbl_author_bio_content)

        about_layout.addStretch()
        about_layout.addWidget(info_box)
        about_layout.addStretch()

        # ==========================================
        # 4. HOW TO USE VIEW (Index 3)
        # Interactive HTML-formatted documentation viewport.
        # ==========================================
        self.how_to_view = QScrollArea()
        self.how_to_view.setWidgetResizable(True)
        self.how_to_view.setStyleSheet("border: none; background-color: #121212;")

        how_to_content = QWidget()
        how_to_layout = QVBoxLayout(how_to_content)
        how_to_layout.setContentsMargins(40, 40, 40, 40)

        # Container populated dynamically by the controller
        self.lbl_how_to_text = QLabel()
        self.lbl_how_to_text.setWordWrap(True)
        self.lbl_how_to_text.setTextFormat(Qt.RichText)
        self.lbl_how_to_text.setStyleSheet("color: #ddd; font-size: 13px; line-height: 160%;")

        how_to_layout.addWidget(self.lbl_how_to_text)
        how_to_layout.addStretch()
        self.how_to_view.setWidget(how_to_content)

        # Register Views 2 and 3
        self.central_stack.addWidget(self.about_view)    
        self.central_stack.addWidget(self.how_to_view)   

        # ==========================================
        # 5. PROCESSING VIEW (Index 4)
        # RaptorScan interface for batch image analysis.
        # ==========================================
        self.processing_view = QWidget()
        proc_layout = QVBoxLayout(self.processing_view)
        proc_layout.setContentsMargins(50, 50, 50, 50)

        # --- Settings Configuration Group ---
        config_group = QGroupBox("Search Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        config_layout.setColumnStretch(1, 1)

        # I/O Path Selection
        self.btn_select_input = QPushButton("📂 Select Input Folder")
        self.lbl_input_path = QLabel("No folder selected")
        config_layout.addWidget(self.btn_select_input, 0, 0)
        config_layout.addWidget(self.lbl_input_path, 0, 1, 1, 3) 

        self.btn_select_output = QPushButton("🎯 Select Output Folder")
        self.lbl_output_path = QLabel("No folder selected")
        config_layout.addWidget(self.btn_select_output, 1, 0)
        config_layout.addWidget(self.lbl_output_path, 1, 1, 1, 3) 

        # Export Toggles
        self.check_export_hm = QCheckBox("Export Heatmaps")
        self.check_copy_orig = QCheckBox("Copy Top Matches")
        config_layout.addWidget(self.check_export_hm, 2, 0)
        config_layout.addWidget(self.check_copy_orig, 2, 1)

        # Engine Optimization Parameters
        self.lbl_proc_thresh = QLabel("Min Score:")
        self.spin_proc_thresh = QDoubleSpinBox()
        self.spin_proc_thresh.setRange(0.0, 1.0)
        self.spin_proc_thresh.setValue(0.70)
        self.spin_proc_thresh.setSingleStep(0.05)
        self.spin_proc_thresh.setFixedWidth(80)

        self.lbl_batch_size = QLabel("Batch Size:")
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 64)
        self.spin_batch_size.setValue(1)
        self.spin_batch_size.setFixedWidth(60)

        # Precise Grid Formatting to prevent widget overlap
        config_layout.addWidget(self.lbl_proc_thresh, 3, 0) 
        config_layout.addWidget(self.spin_proc_thresh, 3, 1, Qt.AlignLeft) 
        config_layout.addWidget(self.lbl_batch_size, 3, 2)  
        config_layout.addWidget(self.spin_batch_size, 3, 3, Qt.AlignLeft) 
        config_layout.setColumnStretch(4, 1)

        proc_layout.addWidget(config_group)

        # --- Execution & Monitoring Area ---
        self.btn_start_proc = QPushButton("START PROCESSING")
        self.btn_start_proc.setFixedHeight(50)
        self.btn_start_proc.setStyleSheet("background-color: #2d5a27; font-weight: bold; font-size: 14px;")
        proc_layout.addWidget(self.btn_start_proc)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("height: 25px; text-align: center;")
        proc_layout.addWidget(self.progress_bar)

        self.proc_logs = QPlainTextEdit()
        self.proc_logs.setReadOnly(True)
        self.proc_logs.setStyleSheet("background-color: #000; color: #0f0; font-family: 'Courier New';")
        proc_layout.addWidget(self.proc_logs)

        # Register final view
        self.central_stack.addWidget(self.processing_view) 

    def _create_menubar(self):
        """
        Initializes the application's top navigation menu bar 
        and defines the actionable items linked in the Controller.
        """
        menubar = self.menuBar()
        
        # Library Operations Menu
        lib_menu = menubar.addMenu("📁 Library")
        self.action_open_folder = lib_menu.addAction("Open images folder")
        lib_menu.addSeparator()
        self.action_new_lib = lib_menu.addAction("New library")
        self.action_open_lib = lib_menu.addAction("Open library")
        self.action_edit_lib = lib_menu.addAction("Manage library")
        lib_menu.addSeparator()

        # Processing Menu
        processing_menu = menubar.addMenu("⚡ RaptorScan")
        self.action_processing = processing_menu.addAction("process images")

        # Documentation Menu
        help_menu = menubar.addMenu("❓ Help")
        self.action_about = help_menu.addAction("About")
        self.action_how_to = help_menu.addAction("How to Use")