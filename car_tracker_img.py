import cv2

# Our Image
img_file = "dataset/car.jpg"

# Our pre-trained car classifier
classifier_file = "cars.xml"

# Create OpenCV image
img = cv2.imread(img_file)

# Convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_n_white, 1.2, 2)

# Draw rectangles around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with cars spotted
cv2.imshow("Car Tracker", img)

# Wait for a keypress
cv2.waitKey()

print("Code Completed")
