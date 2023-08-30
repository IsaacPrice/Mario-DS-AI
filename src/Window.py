from PyQt5.QtWidgets import *

main_analysis = {
    'reward label' : QLabel('Reward: ')
}

windows = {
    "main_analysis" : main_analysis
}

class windowContent:
    def __init__(self, name):
        self.name = name
        self.widgetDict = windows[name]

    
    def update_variables(self, variables):
        self.widgetDict['reward label'].text = 'Reward: ' + variables['reward']



# Slot function to handle button click
class GUI:
    def __init__(self, variables, current_window_name='main_analysis'):
        self.app = QApplication([])
        self.window = QMainWindow()

        self.current_window = windowContent(current_window_name)
        self.current_window.update_variables(variables)

        # Show the window
        self.window.show()

        # Start the event loop
        self.app.exec_()