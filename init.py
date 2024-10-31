import cv2
from PIL import Image
import pillow_heif

def video():
    cap = cv2.VideoCapture(0)  # Use 0 for the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (e.g., convert to grayscale)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def convertHEIFImageToPNG(pathToImage):
    heif_file = pillow_heif.read_heif(pathToImage)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
    image.save("sitting.png", "PNG")


def showImage(img, name):
    # Display the image
    cv2.imshow(name, img)
    cv2.waitKey(0)  # Wait until a key is pressed to close the window
    
def saveImage(img, name):
    cv2.imwrite(name, img)   

# Load an image
# image = cv2.imread("./sitting.png") 
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# resized_image = cv2.resize(image, (300, 300))
# blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
# blurred_image_1 = cv2.GaussianBlur(image, (15, 15), 0)
# blurred_image_2 = cv2.GaussianBlur(image, (15, 15), 5)
# edges = cv2.Canny(image, 100, 200)

# showImage(edges, '1')

video()


 