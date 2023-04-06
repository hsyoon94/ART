import cv2
import os

image_folder = 'inferenced_result'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()