import numpy as np
import cv2

# Create a black image
img = np.zeros((200, 200, 3), dtype=np.uint8)

# Add some text to the image
cv2.putText(img, 'Test Image', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the image
cv2.imwrite('app/static/test_image.jpg', img)
print("Test image created at app/static/test_image.jpg")
