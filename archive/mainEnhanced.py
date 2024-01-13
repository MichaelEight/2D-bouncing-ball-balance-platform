import sys
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import sin, tan, sqrt
import time
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense



def startup():
    update_viewport(None, screenWidth, screenHeight)
    glClearColor(0.5, 0.5, 0.5, 1.0)

def shutdown():
    pass

def key_callback(window, key, scancode, action, mods):
    global startSimulation 
    if key == GLFW_KEY_SPACE and action == GLFW_PRESS:
        reset_simulation()

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
        return True
    return False

def timer_reaches_limit(current_time):
    global time_limit
    # Check if the current time exceeds the specified time limit
    if current_time >= time_limit:
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
    # Reset simulation variables
    ballPosition = [0.0, 0.0]
    ballVelocity = [0.0, 0.0]
    phi = 0.0  # Reset to initial angle or any desired angle
    target_phi = 0.0  # Reset target angle

    # reset time measurement
    local_time = 0.0
    lastTime = glfwGetTime()
    deltaTime = 0.0
    timeSinceLastHit = 0.0

    sim_interation += 1
    showStartingInfo = True

    # Update machine learning model with collected data from the previous simulation
    # Train or update the ML model here using the collected data

    # Restart the simulation
    # Additional code to restart any other necessary variables or components


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

pi = 3.14159
p1 = [0.0, 0.0]
p2 = [0.0, 0.0]
screenHeight = 1000
screenWidth = 1000
platform_a = 0.0
sim_interation = 1
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
time_limit = 60.0
maxFPS = 30.0


transition_duration = 5.0 * 0.5 * maxFPS  # Duration of the transition in seconds
local_time = 0.0
showStartingInfo = True

def render(time):
    global ballAcceleration, ballVelocity, ballPosition
    global phi, target_phi, transition_duration
    global deltaTime, lastTime, timeSinceLastHit, local_time
    global checkForCollision, closeSimulation, screenHeight, screenWidth
    global showStartingInfo, sim_interation

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

    ########################


    # AI/ML goes here


    ########################

    # Check if the ball is outside the screen dimensions
    if (
        ballPosition[0] + ballSize < -screenWidth / 2
        or ballPosition[0] - ballSize > screenWidth / 2
        or ballPosition[1] + ballSize < -screenHeight / 2
        or ballPosition[1] - ballSize > screenHeight / 2
    ):
        print("Ball left the screen. Exiting...")
        closeSimulation = True

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
                ballVelocity = [0.0,0.0]
                ballAcceleration = [0.0,0.0]
                print("Ball stuck!")
                closeSimulation = True

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

    startup()
    while not glfwWindowShouldClose(window) and not closeSimulation:
        # if startSimulation:
            render(glfwGetTime())
            glfwSwapBuffers(window)
            glfwPollEvents()
        # else:
        #     glfwWaitEvents()

    shutdown()
    glfwTerminate()


if __name__ == '__main__':
    main()
