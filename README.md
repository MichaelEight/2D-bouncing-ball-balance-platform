# 2D Bouncing Ball Balancing on the Platform with AI

Using different Python libraries (OpenGL, Tensorflow, NEAT in the future) I am trying to make a working 2D balancing simulator, where a bouncing ball stays as close to target X line as possible.

### Before you give it a go, make sure that...

+ **You have all necessary libraries installed** They are written at the beginning of the file. I'll post a command to install them all soon


## What does each generation do?

### Generation 1 (aka `mainEnhancedPlusML.py`)

Not optimized version. Might have physics issues at large FPS (50+ I suppose). Supports only 1 ball at the time. AI gets ball position, velocity every few frames

### Generation 2 (aka `mainEnhancedPlusMLv2.py`)

Extended reward system to check more things e.g. how long does the ball spend in close proximity to the target

### Generation 3 (aka `mainEnhancedPlusMLv3.py`)

Sorted functions, it's easier to read and manage now. Now rewards data is saved along with the model. You can now simulate multiple pairs (ball & platform) at once!
Simulation got optimized a little and works faster than previous versions. Title of the window displays current iteration and when was the last save.

### Generation 4 (aka `mainEnhancedPlusMLv4.py`)

Redesigned AI, now it collects info about last bounce instead of all the time. Reason - It should be rating the quality of last bounce, how well did the platform do in hitting the ball.

## Issues
+ AI tends to focus on one side (clockwise or anticlockwise turn all the time) and therefore resulting in a quick regression graph

## What is the file structure?
+ saved_modelsv4 - AI models are saved here, as well as reward stats so you can display the trend using `displayGraphv2.py`
+ archive - previous generations are saved there
+ File beginning with `main` and ending with `[number].py` - this is the main file you are supposed to start to train (and later use) the AI
+ You can also use `launchv[number].py` which starts the main file
