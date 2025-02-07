import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import speech_recognition as sr
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for rendering



# Load floor map and process it
def load_floor_map(file_path):
    image = Image.open(file_path).convert("L")  # Convert to grayscale
    floor_map = np.array(image)
    # Assume black pixels (low values) are walls/obstacles and white pixels (high values) are free space
    floor_map = np.where(floor_map < 128, 1, 0)  # 1: obstacle (black), 0: free space (white)
    return floor_map


# Voice command processing
# Voice command processing with 5 seconds limit
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command (5 seconds)...")
        try:
            # Record for up to 5 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio)
            print(f"Command received: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand the command.")
        except sr.RequestError:
            print("Error with the voice recognition service.")
        except sr.WaitTimeoutError:
            print("No voice input detected within the time limit.")
    return None


# Parse the command and calculate movement
def parse_command(command):
    if "forward" in command:
        steps = int(command.split("forward")[-1].strip().split()[0])
        return (-steps, 0)  # Negative row movement
    elif "backward" in command:
        steps = int(command.split("backward")[-1].strip().split()[0])
        return (steps, 0)  # Positive row movement
    elif "left" in command:
        steps = int(command.split("left")[-1].strip().split()[0])
        return (0, -steps)  # Negative column movement
    elif "right" in command:
        steps = int(command.split("right")[-1].strip().split()[0])
        return (0, steps)  # Positive column movement
    return (0, 0)

# Check for obstacles
def is_valid_move(floor_map, position):
    rows, cols = floor_map.shape
    r, c = position
    if 0 <= r < rows and 0 <= c < cols and floor_map[r, c] == 0:
        return True
    return False

# Update function for animation
# Update function for animation
def update_dot(floor_map, position, move, dot):
    # Calculate the new position
    new_position = (position[0] + move[0], position[1] + move[1])

    # Check if the new position is valid
    if is_valid_move(floor_map, new_position):
        dot.set_data([new_position[1]], [new_position[0]])  # Set x and y as lists
        return new_position
    else:
        print("Obstacle ahead!")
        return position  # Return the current position if the move is invalid

# Main simulation
# Main simulation
def simulate_wheelchair_navigation(floor_map_path):
    floor_map = load_floor_map(floor_map_path)
    
    # Calculate the center of the floor map
    rows, cols = floor_map.shape
    position = (rows // 2, cols // 2)  # Center of the image

    # Ensure the initial position is valid
    if not is_valid_move(floor_map, position):
        print("The initial position at the center is invalid (obstructed or out of bounds). Searching for the nearest valid position...")
        found = False
        for radius in range(max(rows, cols)):  # Expand outward from the center
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    new_position = (rows // 2 + dr, cols // 2 + dc)
                    if is_valid_move(floor_map, new_position):
                        position = new_position
                        print(f"Moved initial position to nearest valid point: {position}")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        else:
            print("No valid starting position found. Exiting.")
            return

    fig, ax = plt.subplots()
    ax.imshow(floor_map, cmap="gray")
    dot, = ax.plot(position[1], position[0], "ro")  # Initial dot position

    print("Do you want to use voice control? (yes/no)")
    use_voice = input().strip().lower()

    def on_key(event):
        nonlocal position
        if use_voice == "no":
            if event.key in ["i", "k", "j", "l"]:  # Manual controls
                move = {
                    "i": (-1, 0),  # Up
                    "k": (1, 0),   # Down
                    "j": (0, -1),  # Left
                    "l": (0, 1),   # Right
                }[event.key]
                position = update_dot(floor_map, position, move, dot)
                fig.canvas.draw_idle()

    if use_voice == "yes":
        while True:
            command = get_voice_command()
            if command:
                move = parse_command(command)
                position = update_dot(floor_map, position, move, dot)
                fig.canvas.draw_idle()
    else:
        fig.canvas.mpl_connect("key_press_event", on_key)
        print("Use 'i', 'k', 'j', 'l' to move the dot manually.")

    plt.title("Voice or Manual Control Simulation")
    plt.show()



# Run the simulation with a sample floor map
simulate_wheelchair_navigation(r"C:\Users\meghna\OneDrive\Desktop\walls.jpg")
