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

# Define a simple neural network model using TensorFlow
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # Change input shape to (6,) for 6 features
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer predicting the target angle phi
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize the model
model = create_model()

def ai_agent():
    # Retrieve relevant variables
    ball_pos_x, ball_pos_y = ballPosition
    ball_vel_x, ball_vel_y = ballVelocity
    current_phi = phi
    current_distance_from_x = distance_to_target()

    # Your AI/ML model logic goes here to decide the target phi
    # Prepare the input data for the model
    input_data = [ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, current_phi, current_distance_from_x]

    # Assuming 'model' is your trained neural network
    target_phi = model.predict(np.array([input_data]))[0][0]  # Use your trained model to predict the target angle
    
    return target_phi

# Function to save the model and iteration number
def save_model(model, iteration):
    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Save the model architecture and weights
    model.save(os.path.join(MODEL_DIR, f'model_iteration_{iteration}.h5'))
    
    # Save iteration number to track progress
    with open(os.path.join(MODEL_DIR, 'iteration.txt'), 'w') as file:
        file.write(str(iteration))

# Function to load the model and iteration number
def load_model():
    # Check if the model directory exists
    if not os.path.exists(MODEL_DIR):
        print("No saved models found.")
        return None, None
    
    # Find the latest saved model
    latest_iteration = 0
    latest_model = None
    for file_name in os.listdir(MODEL_DIR):
        if file_name.startswith('model_iteration_') and file_name.endswith('.h5'):
            iteration = int(file_name.split('_')[-1].split('.')[0])
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_model = file_name
    
    if latest_model:
        # Load the latest model
        model_path = os.path.join(MODEL_DIR, latest_model)
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Load the iteration number
        iteration_file = os.path.join(MODEL_DIR, 'iteration.txt')
        with open(iteration_file, 'r') as file:
            loaded_iteration = int(file.read())
        
        return loaded_model, loaded_iteration
    else:
        print("No saved models found.")
        return None, None

# Function to calculate distance between ball and target X
def distance_to_target():
    return abs(ballPosition[0] - targetX)

def startup():
    update_viewport(None, screenWidth, screenHeight)
    glClearColor(0.5, 0.5, 0.5, 1.0)

def shutdown():
    pass

def key_callback(window, key, scancode, action, mods):
    global startSimulation 
    if key == GLFW_KEY_SPACE and action == GLFW_PRESS:
        print()

def calculate_platform_position():
    global p1, p2
    global platform_a
    a = tan(phi * pi / 180) # convert degrees to radians
    platform_a = a

    # x2 = (-2*a*b + sqrt((a*a +1)*L*L - 4*b*b))/(2*(a*a+1))
    x2 = L / (2 * sqrt(a*a + 1)) # Optimization, because b=0
    x1 = -x2

    y2 = a*x2
    y1 = -y2

    p1 = [x1, y1 + b]
    p2 = [x2, y2 + b]

def set_phi(angle):
    global phi
    # -45 <= phi <= 45

    if angle >= 45.0:
        phi = 45.0
    elif angle <= -45.0:
        phi = -45.0
    else:
        phi = angle

# Define the function to smoothly transition between angles
def smooth_transition(current_angle, target_angle, time_elapsed, transition_duration):
    # Calculate the rate of change per second
    rate_of_change = (target_angle - current_angle) / transition_duration
    # Calculate the new angle based on the time elapsed
    new_angle = current_angle + rate_of_change * time_elapsed

    # Ensure the new angle doesn't overshoot the target angle
    if rate_of_change > 0:
        new_angle = min(new_angle, target_angle)
    else:
        new_angle = max(new_angle, target_angle)

    return new_angle

def ball_out_of_bounds_condition():
    global ballPosition, ballSize, screenWidth, screenHeight
    # Check if the ball exceeds the screen boundaries
    if (
        ballPosition[0] + ballSize < -screenWidth / 2
        or ballPosition[0] - ballSize > screenWidth / 2
        or ballPosition[1] + ballSize < -screenHeight / 2
        or ballPosition[1] - ballSize > screenHeight / 2
    ):
        print("Ball left the screen!")
        return True
    return False

def timer_reaches_limit(current_time):
    global time_limit
    # Check if the current time exceeds the specified time limit
    if current_time >= time_limit:
        print("Time ran out!")
        return True
    return False

def check_conditions(time):
    # Check if the ball falls out of the map or the timer reaches the limit
    if (ball_out_of_bounds_condition() or timer_reaches_limit(time)):
        return True
    return False

def reset_simulation():
    global ballPosition, ballVelocity, phi, target_phi
    global local_time, lastTime, deltaTime, timeSinceLastHit
    global sim_interation, showStartingInfo
    global total_distance_covered, simulation_start_time, targetX
    global data, current_iteration

    simulation_end_time = glfwGetTime()
    simulation_time = simulation_end_time - simulation_start_time
    score = total_distance_covered / simulation_time
    print(f"Simulation Time: {simulation_time}, Total Distance: {total_distance_covered}, Score: {score}")

    # Check if it's time to update the model and train it
    if current_iteration >= iterations_to_collect_data:
        update_model_with_collected_data()

    # Reset simulation variables
    ballPosition = [random.uniform(rangeXPos[0], rangeXPos[1]), random.uniform(rangeYPos[0], rangeYPos[1])]
    ballVelocity = [random.uniform(rangeXvel[0], rangeXvel[1]), random.uniform(rangeYvel[0], rangeYvel[1])]
    phi = 0.0  # Reset to initial angle or any desired angle
    target_phi = 0.0  # Reset target angle
    targetX = random.uniform(rangeXTarget[0], rangeXTarget[1])

    print()
    print("Starting pos: ", ballPosition)
    print("Starting vel: ", ballVelocity)
    print("Target X: ", targetX)

    # reset time measurement
    local_time = 0.0
    lastTime = glfwGetTime()
    deltaTime = 0.0
    timeSinceLastHit = 0.0

    # Reset AI variables
    total_distance_covered = 0.0
    simulation_start_time = glfwGetTime()

    sim_interation += 1
    showStartingInfo = True

    # Increment the current iteration count
    current_iteration += 1

def update_model_with_collected_data():
    global data, model, current_iteration

    # Extract features and target values from collected data
    features = np.array([datapoint[:-1] for datapoint in data])
    targets = np.array([datapoint[-1] for datapoint in data])

    # Split data into training and validation sets
    split_index = int(len(features) * 0.8)  # 80-20 split for training-validation
    train_features = features[:split_index]
    train_targets = targets[:split_index]
    val_features = features[split_index:]
    val_targets = targets[split_index:]

    # Train the model
    model.fit(train_features, train_targets, validation_data=(val_features, val_targets), epochs=10, batch_size=32)

    # Evaluate model performance on validation set
    evaluation = model.evaluate(val_features, val_targets)
    print("Evaluation Loss:", evaluation)

    save_model(model, sim_interation)

    # Reset the data container after updating the model
    data = []
    # Reset the current iteration count
    current_iteration = 0

def draw_platform():
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex2f(*p1)
    glVertex2f(*p2)
    glEnd()

def draw_ball(x,y):
    glColor3f(0.0, 1.0, 0.0)
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

    # length = sqrt(normal_vector[0]**2 + normal_vector[1]**2)
    # if length > 0:
    #     normal_vector = [normal_vector[0]/length, normal_vector[1]/length] # normalize the vector
    return normal_vector

def limit_simulation_speed(deltaTime):
    global maxFPS
    target_frame_time = 1.0 / maxFPS  # Calculate the target time for each frame
    
    # Check if the actual frame time exceeds the target frame time
    if deltaTime < target_frame_time:
        sleep_time = target_frame_time - deltaTime
        time.sleep(sleep_time)  # Introduce a delay to limit FPS

def pointsDistance(A, B):
    return sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2)

# Directory to save model files
MODEL_DIR = 'saved_models/'

pi = 3.14159
p1 = [0.0, 0.0]
p2 = [0.0, 0.0]
screenHeight = 1000
screenWidth = 1000
platform_a = 0.0
sim_interation = 0
deltaTime = 0.0
lastTime = 0.0
timeSinceLastHit = 0.0
checkForCollision = True
closeSimulation = False
startSimulation = False  # Flag to control simulation start
lineWidth = 10.0
pointSize = 20.0
target_phi = 0.0


# Starting Settings
ballAcceleration = [0.0, -0.15] # include gravity
ballVelocity     = [0.0, 0.0]
ballPosition     = [0.0, 0.0]
phi = 0.0 
ballSize = 10.0 # it's a square with given a/2 ('radius')
L = 800.0
b = -300.0 # -25.0 was fine
time_limit = 15.0
maxFPS = 60.0

# Ranges of randomness [min, max]
rangeXvel = [-1.5, 1.5]
rangeYvel = [-1.5, 1.5]
rangeXPos = [-100.0, 100.0]
rangeYPos = [-50.0, 50.0]
rangeXTarget = [-60.0, 60.0]

## AI/ML Rewards ##
targetX = 0.0
# calculate avg distance/time as a score
total_distance_covered = 0.0  # Track total distance covered during simulation
simulation_start_time = 0.0   # Track simulation start time
# Define a variable to keep track of the number of iterations to collect data
iterations_to_collect_data = 5
current_iteration = 0
# Define a counter to limit the frequency of model updates or predictions
update_period = 0.1 # Make predictions with breaks
time_since_last_prediction = 0.0

transition_duration = 5.0 * 0.5 * maxFPS  # Duration of the transition in seconds
local_time = 0.0
showStartingInfo = True
data = []  # Store input-output pairs: [x, y, vel_x, vel_y, phi_current, phi_target]

def render(time):
    global ballAcceleration, ballVelocity, ballPosition
    global phi, target_phi, transition_duration
    global deltaTime, lastTime, timeSinceLastHit, local_time
    global checkForCollision, closeSimulation, screenHeight, screenWidth
    global showStartingInfo, sim_interation, data, total_distance_covered, simulation_start_time
    global time_since_last_prediction

    glClear(GL_COLOR_BUFFER_BIT)

    if showStartingInfo:
        print("Current iteration: ", sim_interation)
        showStartingInfo = False        

    if check_conditions(local_time):
        print("Restarting simulation...")
        reset_simulation()

    deltaTime = time - lastTime
    lastTime = time

    local_time += deltaTime
    time_since_last_prediction += deltaTime

    ###########AI/ML#############
    
    distance_to_target_x = distance_to_target()

    # Update the total distance covered during the simulation
    total_distance_covered += distance_to_target_x * deltaTime

    if time_since_last_prediction >= update_period:
        input_data = [ballPosition[0], ballPosition[1], ballVelocity[0], ballVelocity[1], phi, distance_to_target_x]
        target_phi = ai_agent()
        data.append(input_data + [target_phi])

        time_since_last_prediction = 0.0

    ########################

    # Draw line with target_x for visualization
    glLineWidth(2.0)
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex2f(targetX, 300.0)
    glVertex2f(targetX, -300.0)
    glEnd()
    glLineWidth(lineWidth)

    calculate_platform_position()
    draw_platform()

    # update velocity of the ball
    ballVelocity = addVectors(ballVelocity, ballAcceleration)
    # update position of the ball
    ballPosition = addVectors(ballPosition, ballVelocity)

    if(checkForCollision):
        # Check for collision between ball and line
        line_point1 = p1
        line_point2 = p2
        ball_x, ball_y = ballPosition
        ball_radius = ballSize * 1.414

        # Calculate line equation: Ax + By + C = 0
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = line_point2[0] * line_point1[1] - line_point1[0] * line_point2[1]

        # Calculate distance from point to line
        distance = abs(A * ball_x + B * ball_y + C) / sqrt(A * A + B * B)
        distanceFromP1 = pointsDistance(ballPosition, line_point1)
        distanceFromP2 = pointsDistance(ballPosition, line_point2)

        # Check for collision and also if it actually hits the line, not goes by it
        if distance <= ball_radius + lineWidth/2 and distanceFromP1 <= L and distanceFromP2 <= L and local_time - timeSinceLastHit >= 0.5:
            timeSinceLastHit = local_time
            if distance > 0.005:
                normalVector = calculate_normal_vector(platform_a)
                ballVelocity = reflect_velocity(ballVelocity, normalVector)
            else:
                print("Ball stuck!")
                reset_simulation(time)

    # draw ball
    draw_ball(*ballPosition)
    
    glFlush()

    phi = smooth_transition(phi, target_phi, local_time, transition_duration)
    limit_simulation_speed(deltaTime)


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
    global sim_interation
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
    model, sim_interation = load_model()

    # If no model is found, create a new one
    if model is None:
        model = create_model()
        sim_interation = 0

    startup()
    while not glfwWindowShouldClose(window) and not closeSimulation:
        render(glfwGetTime())
        glfwSwapBuffers(window)
        glfwPollEvents()

    shutdown()
    glfwTerminate()


if __name__ == '__main__':
    main()
