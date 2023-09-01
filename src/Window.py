from PyQt5.QtWidgets import *
from data_store import data

def pause_window():
    data['running'] = 0

def continue_window():
    data['running'] = 1

def stop_window():
    data['running'] = -1

# Create the window
app = QApplication([])
window = QMainWindow()

# Create the buttons
pause_button = QPushButton('pause')
continue_button = QPushButton('continue')
stop_button = QPushButton('stop')

