import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a list to store coordinates
coords = []

# Event handler for mouse click
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Append the clicked coordinates to the list
        coords.append((int(event.xdata), int(event.ydata)))
        print(f"Clicked at: {coords[-1]}")
        
        # Display bounding box if 2 points are clicked
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            bbox = [(x1, y1), (x2, y2)]
            print(f"Bounding Box: {bbox}")
            
            # Draw the bounding box on the image
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            plt.draw()

# Load and display the image
def main(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title("Click to get coordinates (2 clicks for a bounding box)")
    
    # Connect the click event to the handler
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# Example usage
# Replace 'your_image.jpg' with the path to your image file
main(r"C:\Users\saisa\OneDrive\Desktop\walls.jpg")