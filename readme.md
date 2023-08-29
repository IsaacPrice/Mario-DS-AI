The AI is fully functional now, but there are still things that need to be done.

 - Change the way the death is determined. Right now, the death is counted when the live goes down, but that doen't help the AI since the animation takes so long. Instead, the death should be determined when the arrow on the bottom goes invisible.
 - Organize the code. Right now, the code is a mess. It needs to be broken up into seperate files, and also make a file that contains all of the constants, such as how many frames in the stack, how many frames to train, and the weights for the reward function.
 - Get it working on Linux 