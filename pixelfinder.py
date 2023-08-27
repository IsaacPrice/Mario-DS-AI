import pyautogui
import time

print("Press Ctrl+C to exit.")

try:
    while True:
        # Get the mouse position
        x, y = pyautogui.position()
        
        # Clear the console screen. For Windows, you can use 'cls' and for macOS and Linux, 'clear'.
        print('\033c', end='')

        # Print the mouse coordinates
        print(f"Mouse Coordinates: X = {x}, Y = {y}")
        
        # Pause for a short time to make it readable
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting...")
