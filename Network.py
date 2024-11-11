import numpy as np
import streamlit as st
import time
from keras.models import Sequential
from keras.layers import Dense

# Define the neural network architecture
input_dim = 7 + 9  # 7 base features + 9 features per surrounding car
model = Sequential()
model.add(Dense(32, input_dim=input_dim, activation='relu', name='input_layer'))
model.add(Dense(32, activation='relu', name='hidden_layer1'))
model.add(Dense(3, activation='softmax', name='output_layer'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')  # Use categorical cross-entropy for classification

# Function to preprocess the input data
def preprocess_input(car, surrounding_cars):
    input_vector = [
        car.location, car.fuel, car.speed, car.acceleration, car.braking, car.steering
    ]

    for surrounding_car in surrounding_cars:
        relative_distance = car.location - surrounding_car.location
        relative_speed = car.speed - surrounding_car.speed
        relative_acceleration = car.acceleration - surrounding_car.acceleration

        input_vector.extend([relative_distance, relative_speed, relative_acceleration])

    # Ensure the input data has a length of 16
    while len(input_vector) < 16:
        input_vector.append(0.0)
    
    return np.array(input_vector)


# Simulated cars with their details
class Car:
    def __init__(self, location, fuel, speed, acceleration, braking, steering):
        self.location = location
        self.fuel = fuel
        self.speed = speed
        self.acceleration = acceleration
        self.braking = braking
        self.steering = steering

    def get_details(self):
        return f"Location: {self.location}, Fuel: {self.fuel}, Speed: {self.speed}, Acceleration: {self.acceleration}, Braking: {self.braking}, Steering: {self.steering}"

# Simulated cars with initial details
cars = [
    Car(12.0, 0.7, 28.0, 0.2, 0.0, 1.0),
    Car(15.0, 0.5, 32.0, 0.0, 0.1, 1.0),
    Car(15.0, 0.4, 32.0, 0.0, 0.1, 1.0)
]

# Gather max values for normalization
max_location = max([car.location for car in cars])
max_fuel = max([car.fuel for car in cars])
max_speed = max([car.speed for car in cars])
max_acceleration = max([car.acceleration for car in cars])
max_braking = max([car.braking for car in cars])
max_steering = max([car.steering for car in cars])

# Create a Streamlit web app
st.set_page_config(layout="wide")  # Set page layout to "wide"
image = "2f6tcqyn.png"
st.image(image, caption=None, width=None, use_column_width=1, clamp=False, channels="RGB", output_format="auto")
st.title("CarLINK: AI Driven Traffic Adviser for Automobiles")

# Initialize a container for the car details and suggested actions
details_container = st.container()

# User input for adding car details
st.sidebar.subheader("Add Car Details")
new_location = st.sidebar.number_input("Location", value=12.0, step=0.1)
new_fuel = st.sidebar.number_input("Fuel", value=0.7, step=0.01)
new_speed = st.sidebar.number_input("Speed", value=28.0, step=0.1)
new_acceleration = st.sidebar.number_input("Acceleration", value=0.2, step=0.01)
new_braking = st.sidebar.number_input("Braking", value=0.0, step=0.01)
new_steering = st.sidebar.number_input("Steering", value=0.0, step=0.01)

def control_traffic(car, surrounding_cars):
    state = preprocess_input(car, surrounding_cars)
    action_probs = model.predict(np.array([state]))[0]
    action = np.argmax(action_probs)

    # Return the suggested traffic control action
    if action == 0:
        return "Accelerate"
    elif action == 1:
        return "Brake"
    elif action == 2:
        return "Maintain speed"

# Button to add a new car
add_button = st.sidebar.button("Add Car")

# Start/Stop Simulation Button
simulation_started = st.sidebar.checkbox("Start Simulation")

def add_new_car(location, fuel, speed, acceleration, braking, steering):
    new_car = Car(location, fuel, speed, acceleration, braking, steering)
    cars.append(new_car)

if add_button:
    add_new_car(new_location, new_fuel, new_speed, new_acceleration, new_braking, new_steering)

# Function to update car details and suggested actions
def update_car_details(cars):
    for i, car in enumerate(cars):
        surrounding_cars = [other_car for j, other_car in enumerate(cars) if j != i]
        
        # Simulate live changes in car parameters
        car.location += np.random.uniform(-1, 1)
        car.fuel -= np.random.uniform(0.01, 0.05)
        car.speed += np.random.uniform(-2, 2)
        car.acceleration += np.random.uniform(-0.1, 0.1)
        car.braking += np.random.uniform(-0.1, 0.1)
        car.steering += np.random.uniform(-0.1, 0.1)

        # Control traffic for the car
        suggested_action = control_traffic(car, surrounding_cars)
        
        # Update the car details and suggested action in the Streamlit app
        with details_container:
            st.write(f"Car Details: {car.get_details()}")
            st.write(f"Suggested Action: {suggested_action}")
            st.write("----")  # Add a separator between car details

# Run the Streamlit app
if simulation_started:  # Simulate continuously
    while True:
        update_car_details(cars)
        time.sleep(1)  # Delay to control the speed of the simulation
        st.experimental_rerun()
