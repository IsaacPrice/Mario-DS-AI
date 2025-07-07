import cv2
import numpy as np

class FrameDisplay:
    def __init__(self, frame_shape=(64, 96), scale=4, spacing=5, window_size=(640, 480), num_q_values=10):
        self.frame_shape = frame_shape
        self.scale = scale
        self.spacing = spacing
        self.window_size = window_size
        self.num_q_values = num_q_values
        self.bar_height = 50  # Height of the bar graph area

    def display_frames(self, array, q_values):
        """
        Display frames from a numpy array and Q-values using OpenCV.

        Args:
        array (np.ndarray): A numpy array of shape (4, height, width).
        q_values (np.ndarray): A numpy array of Q-values.
        """
        if array.shape[1:] != self.frame_shape:
            raise ValueError(f"Each frame in the array must be of shape {self.frame_shape}")
        if len(q_values) != self.num_q_values:
            raise ValueError(f"Number of Q-values must be {self.num_q_values}")

        frames = self._prepare_frames(array)
        resized_frames = [self._resize_frame(frame, self.scale) for frame in frames]
        concatenated_frames = self._concatenate_frames(resized_frames, self.spacing)

        # Resize the Q-values bar to match the width of the concatenated frames
        frame_width = concatenated_frames.shape[1]
        q_values_frame = self._create_q_values_bar(q_values, frame_width)

        # Stack frames and Q-values bar vertically
        final_frame = np.vstack([concatenated_frames, q_values_frame])

        # Centering in the window
        final_frame = self._center_in_window(final_frame, self.window_size)

        cv2.imshow('Frame Stack with Q-values', final_frame)
        cv2.waitKey(1)


    def _prepare_frames(self, array):
        array = array.astype(np.float32)
        array -= array.min()
        array /= array.max()
        array *= 255
        array = array.astype(np.uint8)

        return [frame for frame in array]

    def _resize_frame(self, frame, scale):
        """
        Resize frame with nearest-neighbor interpolation while maintaining aspect ratio.
        """
        height, width = frame.shape
        return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)

    def _concatenate_frames(self, frames, spacing):
        height = max(frame.shape[0] for frame in frames)
        total_width = sum(frame.shape[1] for frame in frames) + spacing * (len(frames) - 1)
        concatenated = np.zeros((height, total_width), dtype=np.uint8)

        current_x = 0
        for frame in frames:
            concatenated[:frame.shape[0], current_x:current_x + frame.shape[1]] = frame
            current_x += frame.shape[1] + spacing

        return concatenated

    def _center_in_window(self, frame, window_size):
        """
        Center frame within the specified window size.
        """
        window_height, window_width = window_size
        frame_height, frame_width = frame.shape[:2]

        vertical_padding = max((window_height - frame_height) // 2, 0)
        horizontal_padding = max((window_width - frame_width) // 2, 0)

        return cv2.copyMakeBorder(frame, vertical_padding, vertical_padding, 
                                  horizontal_padding, horizontal_padding, 
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    def _create_q_values_bar(self, q_values, width):
        """
        Create a bar graph for Q-values.

        Args:
        q_values (np.ndarray): A numpy array of Q-values.
        width (int): The width to resize the Q-values bar to.
        """
        q_values = q_values.cpu().numpy()
        q_values_normalized = cv2.normalize(q_values, None, alpha=0, beta=self.bar_height, norm_type=cv2.NORM_MINMAX)

        bar_width = int(width / self.num_q_values)
        q_values_frame = np.zeros((self.bar_height, width), dtype=np.uint8)

        for i, value in enumerate(q_values_normalized):
            # Draw each bar
            left = i * bar_width
            top = self.bar_height - int(value)
            cv2.rectangle(q_values_frame, (left, top), (left + bar_width, self.bar_height), (255, 255, 255), -1)
            # Label the bar
            cv2.putText(q_values_frame, str(i), (left + 5, self.bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        return q_values_frame

    def close(self):
        cv2.destroyAllWindows()
