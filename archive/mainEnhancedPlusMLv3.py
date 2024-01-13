import sys
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sin, tan, sqrt
import random
import time
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import os

########## WINDOW HANDLER ##########
def startup():
    update_viewport(None, screenWidth, screenHeight)
    glClearColor(0.5, 0.5, 0.5, 1.0)

def shutdown():
    pass

def key_callback(window, key, scancode, action, mods):
    global startSimulation 
    if key == GLFW_KEY_SPACE and action == GLFW_PRESS:
        print()

def update_viewport(window, width, height):
    global screenWidth, screenHeight
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    glMatrixMode(GL_PROJECTION)
    glViewport(0, 0, width, height)
    glLoadIdentity()

    # Calculate half-width and half-height
    half_width = screenWidth/2
    half_height = screenHeight/2

    # Adjust based on aspect ratio
    if width <= height:
        half_width *= width / height
    else:
        half_height *= height / width

    # Set the view
    glOrtho(-half_width, half_width, -half_height, half_height, 1.0, -1.0)
    glTranslatef(0.0, 0.0, 0.0)  # Adjust translation here if needed

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def main():
    global lineWidth, pointSize
    global screenWidth, screenHeight
    global sim_iteration
    if not glfwInit():
        sys.exit(-1)

    window = glfwCreateWindow(screenWidth, screenHeight, __file__, None, None)
    if not window:
        glfwTerminate()
        sys.exit(-1)

    glfwMakeContextCurrent(window)
    glfwSetFramebufferSizeCallback(window, update_viewport)
    glfwSwapInterval(1)
    glfwSetKeyCallback(window, key_callback)

    glLineWidth(lineWidth)
    glPointSize(pointSize)

    # Load the model at the start of the script
    model, sim_iteration = load_model()

    # Check if no model is found
    if model is None:
        print("No saved models found.")
        model = create_model()
        sim_iteration = 0

    startup()
    while not glfwWindowShouldClose(window) and not closeSimulation:
        render(window, glfwGetTime())
        glfwSwapBuffers(window)
        glfwPollEvents()

    shutdown()
    glfwTerminate()


########## DRAWING HANDLER ##########
def calculate_platform_position(phi):
    a = tan(phi * pi / 180) # convert degrees to radians

    x2 = platformLength / (2 * sqrt(a*a + 1))
    x1 = -x2

    y2 = a*x2
    y1 = -y2

    p1 = [x1, y1 + platformVerticalOffset]
    p2 = [x2, y2 + platformVerticalOffset]
    return [p1, p2, a]

def draw_platform(p1, p2, color):
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_LINES)
    glVertex2f(*p1)
    glVertex2f(*p2)
    glEnd()

def draw_ball(x, y, color):
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex2f(x - ballSize,y - ballSize)
    glVertex2f(x + ballSize,y - ballSize)
    glVertex2f(x + ballSize,y + ballSize)
    glVertex2f(x - ballSize,y + ballSize)
    glEnd()

    # Second set of quads (rotated by 45 degrees)
    glBegin(GL_QUADS)
    rotated_ball_size = ballSize * 1.4142  # To maintain the size for rotated squares
    glVertex2f(x - rotated_ball_size, y)  # Rotate by 45 degrees
    glVertex2f(x, y - rotated_ball_size)
    glVertex2f(x + rotated_ball_size, y)
    glVertex2f(x, y + rotated_ball_size)
    glEnd()


########## PHYSICS & SIMULATION HANDLER ##########
# Define the function to smoothly transition between angles
def smooth_transition(current_angle, target_angle):
    # Calculate the rate of change per second
    rate_of_change = (target_angle - current_angle) / transition_duration
    # Calculate the new angle based on the time elapsed
    new_angle = current_angle + rate_of_change * local_time

    # Ensure the new angle doesn't overshoot the target angle
    if rate_of_change > 0:
        new_angle = min(new_angle, target_angle)
    else:
        new_angle = max(new_angle, target_angle)

    return new_angle

def ball_out_of_bounds_condition(ballPosition):
    global ballSize, screenWidth, screenHeight

    # Check if the ball exceeds the screen boundaries
    if (
        ballPosition[0] + ballSize < -screenWidth / 2
        or ballPosition[0] - ballSize > screenWidth / 2
        or ballPosition[1] + ballSize < -screenHeight / 2
        or ballPosition[1] - ballSize > screenHeight / 2
    ):
        return True
    return False

def timer_reaches_limit():
    global local_time, time_limit, frames_in_round, time_limit_by_frames
    # Check if the current time exceeds the specified time limit
    if frames_in_round >= time_limit_by_frames:
        print("Time ran out")
        return True # Time ran out

    # if local_time >= time_limit: 
    #     print("Time ran out")
    #     return True # Time ran out
    return False

def all_balls_out():
    global frames_in_round_checks

    # if time_since_last_check >= checks_update_period: # Slow down check frequency
    if frames_in_round_checks >= checks_update_period_by_frames: # Slow down check frequency
        frames_in_round_checks = 0
        for i in range(number_of_simulated_pairs):
            if pair_instances[i].isAlive:
                return False # At least one alive
        print("All balls eliminated")
        return True
    else:
        return False

def check_conditions():
    # Check if the ball falls out of the map or the timer reaches the limit
    if (all_balls_out() or timer_reaches_limit()):
        return True
    return False

def addVectors(u, v):
    return [u[0] + v[0], u[1] + v[1]]

def reflect_velocity(v, n):
    dot_product = 2 * (v[0] * n[0] + v[1] * n[1])  # Dot product of v and n
    mag_squared = n[0] ** 2 + n[1] ** 2
    reflected_v = [
        v[0] - (dot_product / mag_squared) * n[0],
        v[1] - (dot_product / mag_squared) * n[1],
    ]  # Calculate reflected vector
    return reflected_v

def calculate_normal_vector(a):
    if a != 0:
        normal_vector = [-a, 1]  # If a is not zero, the normal vector is [-1/a, 1]
    else:
        normal_vector = [0.0, 1.0]  # For a horizontal line (a = 0), the normal vector is [1, 0]

    return normal_vector

def limit_simulation_speed(deltaTime):
    global maxFPS
    target_frame_time = 1.0 / maxFPS  # Calculate the target time for each frame
    
    # Check if the actual frame time exceeds the target frame time
    if deltaTime < target_frame_time:
        sleep_time = target_frame_time - deltaTime
        time.sleep(sleep_time)  # Introduce a delay to limit FPS

def pointsDistance(A, B):
    return sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 )


########## AI HANDLER ##########
# Define a simple neural network model using TensorFlow
def create_model():
    model = tf.keras.Sequential([
        # Attempt to reduce overfitting by reducing number of layers
        tf.keras.layers.Dense(6, activation='relu', input_shape=(6,)), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer predicting the target angle phi
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize the model
model = create_model()

# def ai_agent(ballPosition, ballVelocity, current_phi, current_distance_from_x):
#     # Retrieve relevant variables
#     ball_pos_x, ball_pos_y = ballPosition
#     ball_vel_x, ball_vel_y = ballVelocity

#     # Prepare the input data for the model
#     input_data = [ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, current_phi, current_distance_from_x]

#     # AI/ML model logic decides the target phi
#     target_phi = model.predict(np.array([input_data]))[0][0]  # Use your trained model to predict the target angle

#     return target_phi

# Version of AI agent redone by GPT4
def ai_agent(ballPosition, ballVelocity, current_phi, current_distance_from_x):
    # Prepare the input data for the model
    raw_input_data = np.array([ballPosition[0], ballPosition[1], ballVelocity[0], ballVelocity[1], current_phi, current_distance_from_x])
    normalized_input_data = input_normalizer.transform(raw_input_data)

    # AI/ML model logic decides the target phi
    target_phi = model.predict(normalized_input_data.reshape(1, -1))[0][0]  # Reshape data for prediction

    # Clip the target_phi to the [-limit_phi, limit_phi] range
    target_phi = np.clip(target_phi, -limit_phi, limit_phi)

    return target_phi


def save_model(model, iteration, data):
    global average_rewards

    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Save the model architecture and weights
    model.save(os.path.join(MODEL_DIR, f'model_iteration_{iteration}.h5'))
    
    # Save iteration number to track progress
    with open(os.path.join(MODEL_DIR, 'iteration.txt'), 'w') as file:
        file.write(str(iteration))
    
    # Save the data used for training
    np.save(os.path.join(MODEL_DIR, f'data_iteration_{iteration}.npy'), data)

    # Save the average rewards
    avg_rewards_path = os.path.join(MODEL_DIR, f'average_rewards_iteration_{iteration}.npy')
    np.save(avg_rewards_path, np.array(average_rewards))

def load_model():
    global last_save_at_iteration
    # Check if the model directory exists
    if not os.path.exists(MODEL_DIR):
        print("No saved models found.")
        return None, None  # Return None for both model and iteration number
    
    # Find the latest saved model
    latest_iteration = 0
    latest_model = None
    for file_name in os.listdir(MODEL_DIR):
        if file_name.startswith('model_iteration_') and file_name.endswith('.h5'):
            iteration = int(file_name.split('_')[-1].split('.')[0])
            if iteration > latest_iteration:
                latest_iteration = iteration
                last_save_at_iteration = iteration
                latest_model = file_name
    
    if latest_model:
        global average_rewards

        # Load the latest model
        model_path = os.path.join(MODEL_DIR, latest_model)
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Read the iteration number from the iteration file
        iteration_file = os.path.join(MODEL_DIR, 'iteration.txt')
        with open(iteration_file, 'r') as file:
            loaded_iteration = int(file.read())
        
       # Load average rewards if available
        avg_rewards_path = os.path.join(MODEL_DIR, f'average_rewards_iteration_{latest_iteration}.npy')
        if os.path.exists(avg_rewards_path):
            average_rewards = list(np.load(avg_rewards_path))

        return loaded_model, loaded_iteration
    else:
        print("No saved models found.")
        return None, None

def update_model_with_rewards():
    global data, model, current_iteration

    # Extract features and target values from collected data
    features = np.array([datapoint[:-1] for datapoint in data])
    targets = np.array([datapoint[-1] for datapoint in data])

    split_index = int(len(features) * 0.8)  # 80-20 split for training-validation
    train_features = features[:split_index]
    train_targets = targets[:split_index]
    val_features = features[split_index:]
    val_targets = targets[split_index:]

    # Train the model with the collected data
    model.fit(train_features, train_targets, validation_data=(val_features, val_targets), epochs=10, batch_size=32)

    # Evaluate model performance on validation set
    evaluation = model.evaluate(val_features, val_targets)
    print("Evaluation Loss:", evaluation)

    # Reset the data container after updating the model
    data = []
    # Reset the current iteration count
    current_iteration = 0

class Normalizer:
    def __init__(self, min_vals, max_vals):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
    
    def transform(self, data):
        return (data - self.min_vals) / (self.max_vals - self.min_vals)

    def inverse_transform(self, data):
        return (data * (self.max_vals - self.min_vals)) + self.min_vals

class BallPlatformPair:
    def __init__(self, id, initial_position = [0.0, 0.0], initial_velocity = [0.0, 0.0], initial_acceleration = [0.0, 0.0]):
        # Common, Simulation, Stats
        self.color = [random.random(), random.random(), random.random()]
        self.id = id
        self.timeSinceLastHit = 0.0
        self.total_distance_covered = 0.0
        self.targetX = 0.0
        self.isAlive = True
        self.isDisplayed = True
        self.total_reward = 0.0
        self.reward_rate = 0.0

        # Ball
        self.ballAcceleration = initial_acceleration
        self.ballVelocity = initial_velocity
        self.ballPosition = initial_position
        self.time_elapsed = 0.0
        self.previous_distance_to_target_x = self.distance_to_target()

        # Platform
        self.platform_a = 0.0
        self.phi = 0.0
        self.target_phi = 0.0
        self.p1 = [0.0, 0.0]
        self.p2 = [0.0, 0.0]

    def render(self):
        if(self.isDisplayed):
            # Render the ball and platform
            draw_platform(self.p1, self.p2, self.color)
            draw_ball(self.ballPosition[0], self.ballPosition[1], self.color)        

    def randomize(self):
        global gravity

        self.ballAcceleration = [0.0, gravity]
        self.ballVelocity = [random.uniform(rangeXvel[0], rangeXvel[1]), random.uniform(rangeYvel[0], rangeYvel[1])]
        self.ballPosition = [random.uniform(rangeXpos[0], rangeXpos[1]), random.uniform(rangeYPos[0], rangeYPos[1])]
        self.targetX = random.uniform(rangeXTarget[0], rangeXTarget[1])

    def reset(self):
        self.phi = 0.0  # Reset to initial angle or any desired angle
        self.target_phi = 0.0  # Reset target angle
        self.total_distance_covered = 0.0
        self.timeSinceLastHit = 0.0
        self.time_elapsed = 0.0
        self.p1, self.p2, self.platform_a = calculate_platform_position(self.phi)
        self.isAlive = True
        self.isDisplayed = True

        self.randomize()

    def getData(self):
        return [self.ballPosition[0], self.ballPosition[1], self.ballVelocity[0],
                self.ballVelocity[1], self.phi, self.distance_to_target()]

    def distance_to_target(self):
        return abs(self.targetX - self.ballPosition[0])

    def die(self, cause):
        print("Eliminated because: ", cause)
        self.isAlive = False
        self.isDisplayed = False
        if self.time_elapsed > 0:  # To avoid division by zero
            self.reward_rate = self.total_reward / self.time_elapsed
        else:
            self.reward_rate = 0

    def updateLogic(self, deltaTime):
        global total_reward_in_round
        distance_to_target_x = self.distance_to_target()

        # Update the total distance covered during the simulation
        self.total_distance_covered += distance_to_target_x * deltaTime # self note, why is it *deltaTime exactly?? (TODO mark to spot)

        # Calculate reward based on the current state
        reward = calculate_reward(self.phi, *self.ballPosition, *self.ballVelocity, *self.p1,
                                    *self.p2, distance_to_target_x, self.previous_distance_to_target_x)
        total_reward_in_round += reward
        self.total_reward += reward

        input_data = self.getData()
        self.target_phi = ai_agent(self.ballPosition, self.ballVelocity, self.phi, distance_to_target_x)
        # Constrain the target_phi within the desired range
        self.target_phi = max(min(self.target_phi, limit_phi), -limit_phi)
        data.append(input_data + [reward])

        self.previous_distance_to_target_x = distance_to_target_x

        if self.isAlive:
            self.time_elapsed += AIML_update_period

    def updateGraphics(self):
        p1, p2 = self.p1, self.p2

        if self.isAlive:
            if ball_out_of_bounds_condition(self.ballPosition):
                self.die("out of bounds")
                print("Ball left the screen!")
                return

            # Update velocity of the ball
            self.ballVelocity = addVectors(self.ballVelocity, self.ballAcceleration)
            # Update position of the ball
            self.ballPosition = addVectors(self.ballPosition, self.ballVelocity)

            if(checkForCollision):
                # Check for collision between ball and line
                ball_x, ball_y = self.ballPosition
                ball_radius = ballSize * 1.414

                # Calculate line equation: Ax + By + C = 0
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = p2[0] * p1[1] - p1[0] * p2[1]

                # Calculate distance from point to line
                if A*A + B*B != 0:
                    distance = abs(A * ball_x + B * ball_y + C) / sqrt(A * A + B * B)
                else:
                    distance = 0.0 
                distanceFromP1 = pointsDistance(self.ballPosition, p1)
                distanceFromP2 = pointsDistance(self.ballPosition, p2)

                # Check for collision and also if it actually hits the line, not goes by it
                if (distance <= ball_radius + lineWidth/2 and distanceFromP1 <= platformLength
                    and distanceFromP2 <= platformLength and local_time - self.timeSinceLastHit >= 0.5):

                    self.timeSinceLastHit = local_time
                    if distance > 0.005:
                        normalVector = calculate_normal_vector(self.platform_a)
                        self.ballVelocity = reflect_velocity(self.ballVelocity, normalVector)
                    else:
                        self.die("Stuck in the platform")
                        return
            

            self.phi = smooth_transition(self.phi, self.target_phi)
            self.p1, self.p2, self.platform_a = calculate_platform_position(self.phi)
    


########## SIMULATION MAIN PART ##########

def reset_simulation():
    global local_time, lastTime, deltaTime, simulation_start_time, frames_in_round, frames_in_round_AIML
    global data, sim_iteration, current_iteration, showStartingInfo, average_rewards, avg_reward_rate
    global last_save_at_iteration

    # Check if it's time to update the model and train it
    if current_iteration >= iterations_to_collect_data and enableLearning:
        update_model_with_rewards()
        save_model(model, sim_iteration, data)
        last_save_at_iteration = sim_iteration

    # Calculate average reward rate instead of total reward
    total_reward_rate = sum([instance.reward_rate for instance in pair_instances if instance.time_elapsed > 0])
    avg_reward_rate = total_reward_rate / number_of_simulated_pairs if number_of_simulated_pairs > 0 else 0
    average_rewards.append(avg_reward_rate)
    print("Average reward rate per instance this round: ", avg_reward_rate)

    # Reset time measurement
    local_time = 0.0
    lastTime = glfwGetTime()
    deltaTime = 0.0
    frames_in_round = 0.0
    frames_in_round_AIML = 0.0

    showStartingInfo = True

    # Reset AI variables
    simulation_start_time = glfwGetTime()
    sim_iteration += 1

    # Increment the current iteration count
    current_iteration += 1

# New reward function v3 suggested by GPT4 
def calculate_reward(angle, ballX, ballY, velX, velY, p1X, p1Y, p2X, p2Y, distance_to_target_x, previous_distance_to_target_x):
    # Constants for reward calculation
    distance_weight = -0.05  # Penalize based on distance
    stability_reward = 5.0  # Reward for keeping the ball stable
    angle_penalty = -5.0  # Penalize the platform's angle that increases distance
    ball_loss_penalty = -200.0  # Large penalty for losing the ball
    approach_reward = 1.0  # Reward for decreasing distance to target X
    sustained_center_reward = 0.5  # Reward for staying in the "good" zone
    good_zone_threshold = 5

    # Reward for approaching the target
    if distance_to_target_x < previous_distance_to_target_x:
        approach_reward = approach_reward * (previous_distance_to_target_x - distance_to_target_x)

    # Reward for staying near the target X
    if abs(ballX - targetX) < good_zone_threshold:  # good_zone_threshold defines what is considered near
        sustained_reward = sustained_center_reward
    else:
        sustained_reward = 0

    # Calculate distance reward
    distance_reward = distance_weight * distance_to_target_x

    # Reward for reducing velocity towards the center
    velocity_reward = stability_reward * -abs(velX) if (ballX > 0 and velX < 0) or (ballX < 0 and velX > 0) else -abs(velX)

    # Calculate angle penalty
    if (ballX > 0 and angle > 0) or (ballX < 0 and angle < 0):
        angle_reward = angle_penalty * abs(angle)
    else:
        angle_reward = 0

    # Calculate ball loss penalty
    ball_loss_reward = ball_loss_penalty if ball_out_of_bounds_condition([ballX, ballY]) else 0

    # Total reward
    total_reward = (distance_reward + velocity_reward + angle_reward + 
                    ball_loss_reward + approach_reward + sustained_reward)

    return total_reward



# Directory to save model files
MODEL_DIR = 'saved_modelsv3/'

## WINDOW, MATH, SIMULATION HIDDEN VARS
pi = 3.14159
screenHeight = 1000
screenWidth = 1000
sim_iteration = 0
local_time = 0.0
deltaTime = 0.0
lastTime = 0.0
frames_in_round = 0.0
frames_in_round_AIML = 0.0
frames_in_round_checks = 0.0
closeSimulation = False
startSimulation = False  # Flag to control simulation start
data = []  # Store input-output pairs: [x, y, vel_x, vel_y, phi_current, phi_target]
last_save_at_iteration = 0

## DRAWING PARAMS
lineWidth = 10.0
pointSize = 20.0
ballSize = 10.0 # it's a square with given a/2 ('radius')
platformLength = 800.0
platformVerticalOffset = -300.0 # -25.0 was fine

## SIMULATION PARAMS
showStartingInfo = True
checkForCollision = True
gravity = -0.15
maxFPS = 60.0
time_limit = 25.0
time_limit_by_frames = time_limit * maxFPS
transition_duration = 5.0 * 0.5 * maxFPS  # Duration of the transition in seconds
AIML_update_period = 0.5 # Limit predictions frequency
AIML_update_period_by_frames = AIML_update_period * maxFPS
checks_update_period = 0.25
checks_update_period_by_frames = checks_update_period * maxFPS

# Define the minimum and maximum values for each feature
min_vals = [-500, -300, -5, -10, -10, -500]
max_vals = [500, 100, 5, 10, 10, 500]

# Create an instance of the Normalizer class with these values
input_normalizer = Normalizer(min_vals, max_vals)

## DEFAULT VALUES AND RANGES [min, max]
limit_phi = 10.0 
rangeXvel = [-1.0, 1.0]
rangeYvel = [0.0, 0.0]
rangeXpos = [-20.0, 20.0]
rangeYPos = [0.0, 0.0]
rangeXTarget = [0.0, 0.0]
enableLearning = True
iterations_to_collect_data = 10
number_of_simulated_pairs = 5

## AI/ML Rewards
targetX = 0.0
total_distance_covered = 0.0  # Track total distance covered during simulation
simulation_start_time = 0.0   # Track simulation start time
current_iteration = 0
time_since_last_prediction = 0.0
time_since_last_check = 0.0
total_reward_in_round = 0.0 # for average, total/num_of_sim_pairs
avg_reward_rate = 0.0
average_rewards = []

pair_instances = []
for i in range(number_of_simulated_pairs):
    instance = BallPlatformPair(id=i)
    instance.reset()
    pair_instances.append(instance)

def render(window, time):
    global transition_duration
    global deltaTime, lastTime, local_time, simulation_start_time, frames_in_round, frames_in_round_AIML, frames_in_round_checks
    global checkForCollision, closeSimulation, screenHeight, screenWidth
    global showStartingInfo, sim_iteration, data 
    global time_since_last_prediction, total_reward_in_round, time_since_last_check
    global last_save_at_iteration

    # Update the window title with the current iteration
    title = f"Simulation - Iteration: {sim_iteration} - Last Saved At: {last_save_at_iteration}"
    glfwSetWindowTitle(window, title)

    glClear(GL_COLOR_BUFFER_BIT)

    if showStartingInfo:
        print("Current sim_iteration: ", sim_iteration)
        showStartingInfo = False        

    if check_conditions():
        print("Average reward per instance this round: ", total_reward_in_round / number_of_simulated_pairs)
        reset_simulation()
        for i in range(number_of_simulated_pairs):
            pair_instances[i].reset()

    ## Time Calculations ##
    deltaTime = time - lastTime
    lastTime = time

    local_time += deltaTime


    ########### Actual Simulation #############

    if frames_in_round_AIML >= AIML_update_period_by_frames: # Measure by frames, not time
        for i in range(number_of_simulated_pairs):
            pair_instances[i].updateLogic(deltaTime)
        frames_in_round_AIML = 0.0

    # Update Physics
    for i in range(number_of_simulated_pairs):
        pair_instances[i].updateGraphics()
        pair_instances[i].render() # Render right away

    ########################
    
    glFlush()
    # limit_simulation_speed(deltaTime)
    frames_in_round += 1
    frames_in_round_AIML += 1
    frames_in_round_checks += 1


if __name__ == '__main__':
    main()