from PyQt5.QtWidgets import *
import numpy as np
#from data_store import data

def main_window(data):
    # Create the window
    app = QApplication([])
    window = QMainWindow()

    # Create box
    vbox = QGridLayout()

    # Add Widgets (specifying rows and columns)
    vbox.addWidget(pause_button, 0, 0)
    vbox.addWidget(continue_button, 0, 1)
    vbox.addWidget(stop_button, 0, 2)

    # Create a central widget and set the layout
    central_widget = QWidget()
    central_widget.setLayout(vbox)

    window.setCentralWidget(central_widget)

    window.show()

    app.exec_()

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

        # Create box
        vbox = QGridLayout()

        # Add Widgets (specifying rows and columns)
        vbox.addWidget(self.none_label, 0, 0)
        vbox.addWidget(self.left_label, 1, 0)
        vbox.addWidget(self.right_label, 2, 0)
        vbox.addWidget(self.run_left_label, 3, 0)
        vbox.addWidget(self.run_right_label, 4, 0)
        vbox.addWidget(self.jump_label, 5, 0)
        vbox.addWidget(self.jump_left_label, 6, 0)
        vbox.addWidget(self.jump_right_label, 7, 0)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(vbox)

        self.window.setCentralWidget(central_widget)

        self.window.show()
    
    def update_labels(self, list):
        try:
            self.none_label.setText('None: ' + str(np.array(list)[0]))
            self.left_label.setText('Walk Left: ' + str(np.array(list)[1]))
            self.right_label.setText('Walk Right: ' + str(np.array(list)[2]))
            self.run_left_label.setText('Run Left: ' + str(np.array(list)[3]))
            self.run_right_label.setText('Run Right: ' + str(np.array(list)[4]))
            self.jump_label.setText('Jump: ' + str(np.array(list)[5]))
            self.jump_left_label.setText('Jump Left: ' + str(np.array(list)[6]))
            self.jump_right_label.setText('Jump Right: ' + str(np.array(list)[7]))

            self.window.show()
        except:
            pass # Not really sure, but every few times, it would give a nonsense number






