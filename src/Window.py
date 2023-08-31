from PyQt5.QtWidgets import *


# Slot function to handle button click
class GUI:
    def __init__(self, variables, current_window_name='main_analysis'):
        self.app = QApplication([])
        self.window = QMainWindow()

        self.window.setWindowTitle('Info')
        self.window.setGeometry(100, 100, 500, 300)

        self.widgets = {
            'reward_label' : QLabel('Reward: ' + str(variables['Reward']))
        }

        # Show the window
        self.window.show()

        # Start the event loop
        self.app.exec_()
    
    def update_variables(self, variables):
        self.widgets['reward_label'].setText('Reward: ' + str(variables['Reward']))
