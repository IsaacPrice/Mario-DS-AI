import os
import numpy as np
import curses

action_mapping = {
    0 : "1 - Nothing: ",
    1 : "2 - Walk Left: ",
    2 : "3 - Walk Right: ",
    3 : "4 - Run Left: ",
    4 : "5 - Run Right: ",
    5 : "6 - Jump: ",
    6 : "7 - Jump Left: ",
    7 : "8 - Jump Right: "
}

class Dashboard:
    def __init__(self, max_height=10):
        self.max_height = max_height
        self.prev_actions = None
        self.prev_q_values = None

    def scale_values(self, values):
        try:
            self.max_value = 50
            if self.max_value <= 0:
                return np.zeros_like(values)
            return np.round((values / self.max_value) * self.max_height).astype(int)
        except Exception as e:
            print(f"Error in scaling values: {e}")
            return []
    
    def generate_row(self, actions, i, max_action):
        row = []
        for val in actions:
            if val == max_action and val >= i:
                row.append("\033[92m█\033[0m")  # Green
            elif val >= i:
                row.append("\033[97m█\033[0m")  # White
            elif i <= 1 and val <= 10:
                row.append("_")  # underscore for zero or less
            else:
                row.append(" ")  # blank
        return " ".join(row)

    def update(self, data):
        
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            actions = self.scale_values(data.get('actions', []))
            actions_raw = data.get('actions', [])
            q_values = self.scale_values(data.get('q_values', []))

            max_action = np.max(actions)
            max_q_value = np.max(q_values)

            if actions is not None and q_values is not None:
                print("Actions and Q-Values:")
                for i in range(self.max_height, 0, -1):
                    action_row = self.generate_row(actions, i, max_action)
                    q_value_row = self.generate_row(q_values, i, max_q_value)
                    #exact = action_mapping[i] + str(actions_raw[i])
                    print(f"{action_row}  |  {q_value_row}")
                    
            
            self.prev_actions = actions
            self.prev_q_values = q_values

        except Exception as e:
            os.system('cls' if os.name == 'nt' else 'clear')
            actions = self.prev_actions
            actions_raw = data.get('actions', [])
            q_values = self.prev_q_values

            max_action = np.max(actions)
            max_q_value = np.max(q_values)

            if actions is not None and q_values is not None:
                print("Actions and Q-Values:")
                for i in range(self.max_height, 0, -1):
                    action_row = self.generate_row(actions, i, max_action)
                    q_value_row = self.generate_row(q_values, i, max_q_value)
                    '''exact = action_mapping[i + 1] + str(actions_raw[i])
                    if i <= 7:
                        print(f"{action_row}  |  {q_value_row}   {exact}")
                    else: '''
                    print(f"{action_row}  |  {q_value_row}")
        
        print('1 2 3 4 5 6 7 8  |  1 2 3 4 5 6 7 8')

        print("\nAdditional Info:")
        print(f"Velocity: {data.get('velocity', 'N/A')}")
        print(f"Reward: {data.get('reward', 'N/A')}")
        print(f"Total Reward: {data.get('total_reward', 'N/A')}")
        print(f"Level: {data.get('level', 'N/A')}")


