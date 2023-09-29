from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import numpy as np

class Window:
    def __init__(self):
        # Create the window
        self.app = QApplication([])
        self.window = QMainWindow()

        # Edit the window settings
        self.window.setGeometry(50, 50, 100, 100)
        self.window.setWindowTitle("Info")

        # Create widgets
        self.none_label = QLabel('None: ')
        self.left_label = QLabel('Walk Left: ')
        self.right_label = QLabel('Walk Right: ')
        self.run_left_label = QLabel('Run Left: ')
        self.run_right_label = QLabel('Run Right: ')
        self.jump_label = QLabel('Jump: ')
        self.jump_left_label = QLabel('Jump Left: ')
        self.jump_right_label = QLabel('Jump Right: ')

        # Create the bars
        name_bar_in_order = ['none_bar', 'left_bar', 'right_bar', 'run_left_bar', 'run_right_bar', 'jump_bar', 'jump_left_bar', 'jump_right_bar']
        self.bars = {}
        for name in name_bar_in_order:
            self.bars[name] = QProgressBar()
            self.bars[name].setMaximumHeight(100)
            self.bars[name].setOrientation(Qt.Vertical)
            self.bars[name].setFixedHeight(100)
            self.bars[name].setTextVisible(False)
            self.bars[name].setStyleSheet("""QProgressBar {
                                            border: 2px solid grey;
                                            text-align: center;
                                        }
                                        QProgressBar::chunk {
                                            background-color: blue;
                                        }""")

        # Create box
        vbox = QGridLayout()

        # Add Widgets (specifying rows and columns)
        vbox.addWidget(self.none_label, 1, 0)
        vbox.addWidget(self.left_label, 1, 1)
        vbox.addWidget(self.right_label, 1, 2)
        vbox.addWidget(self.run_left_label, 1, 3)
        vbox.addWidget(self.run_right_label, 1, 4)
        vbox.addWidget(self.jump_label, 1, 5)
        vbox.addWidget(self.jump_left_label, 1, 6)
        vbox.addWidget(self.jump_right_label, 1, 7)

        count = 0
        for name, value in self.bars.items():
            vbox.addWidget(value, 0, count)
            count += 1

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(vbox)

        self.window.setCentralWidget(central_widget)

        self.window.show()
    
    def update_labels(self, list):
        try:
            self.none_label.setText('None: ' + str(round(np.array(list)[0], 1)))
            self.left_label.setText('Walk Left: ' + str(round(np.array(list)[1], 1)))
            self.right_label.setText('Walk Right: ' + str(round(np.array(list)[2], 1)))
            self.run_left_label.setText('Run Left: ' + str(round(np.array(list)[3], 1)))
            self.run_right_label.setText('Run Right: ' + str(round(np.array(list)[4], 1)))
            self.jump_label.setText('Jump: ' + str(round(np.array(list)[5], 1)))
            self.jump_left_label.setText('Jump Left: ' + str(round(np.array(list)[6], 1)))
            self.jump_right_label.setText('Jump Right: ' + str(round(np.array(list)[7], 1)))

            count = 0
            for key, value in self.bars.items():
                value.setValue(int(np.array(list)[count]))
                value.repaint()
                count += 1

            self.app.processEvents()
            self.window.show()
        except:
            pass # Not really sure, but every few times, it would give a nonsense number






