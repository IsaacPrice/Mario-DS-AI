MARIO AI Using Deep Q-Networks
==============================
### Overview
This repository houses an AI agent designed to learn and play the Mario game. Utilizing Deep Q-Networks (DQN), the agent learns optimal policies for navigating through game levels. The project aims to explore various AI techniques and offer an extensible platform for further experimentation.
### Features
* Real-time learning through DQN
* GUI for hyperparameter tuning and real-time performance monitoring
* Modular architecture for easy experimentation
### Prerequisites
* Python 3.11
* Tensorflow 2.13
* Py-desmume Emulator
### Installation
1. <p>Clone the repository: <br><code>git clone https://github.com/yourusername/MarioAI.git</code></p>
2. <p>Install the dependencies: <br><code>pip install -r requirements.txt</code></p>
### Usage
1. Add in the Mario DS rom named 'NSMB.ds' inside the main directory
2. <p>Run the main loop:<br><code>python Main.py</code></p>
### Configuration
The GUI allows you to:
* Start, pause, and stop training
* Modify hyperparameters such as learning rate ('**alpha**'), discount factor ('**gamma**'), and exploration rate ('**epsilon**')
* Monitor real-time performance metrics
* Save and load models
### Contributing
Feel free to fork the repository and submit pull requests for any enhancements.
### License
This project is under the MIT License - see the LICENSE.md file for details.
