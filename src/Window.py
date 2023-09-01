from PyQt5.QtWidgets import *
#from data_store import data

def main_window(data):
    def pause_window():
        data['running'] = 0
        print(data['running'])

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

    # Bind the buttons to their functions
    pause_button.clicked.connect(pause_window)
    continue_button.clicked.connect(continue_window)
    stop_button.clicked.connect(stop_window)

    # Edit the window settings
    window.setGeometry(50, 50, 100, 100)
    window.setWindowTitle("Info")

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

if __name__ == '__main__':
    pass