def main():

    import cv2

    # Our Video
    video = cv2.VideoCapture("dataset/video2.mp4")

    # Our pre-trained car & pedestrian classifier
    classifier_cars = "cars.xml"
    classifier_peds = "haarcascade_fullbody.xml"

    # Create a car & pedetrian classifier
    car_tracker = cv2.CascadeClassifier(classifier_cars)
    ped_tracker = cv2.CascadeClassifier(classifier_peds)

    while True:

        # Read the current frame
        (read_successful, frame) = video.read()

        # Safe coding
        if read_successful:

            # Must convert to grayscale
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Detect cars & pedestrians
        cars = car_tracker.detectMultiScale(
            image=grayscaled_frame, scaleFactor=1.05, minNeighbors=2, minSize=(120, 120))
        peds = ped_tracker.detectMultiScale(
            image=grayscaled_frame, scaleFactor=1.05, minNeighbors=2, minSize=(120, 120))

        # Draw rectangles around cars
        loop(frame, cars, (0, 0, 255))
        loop(frame, peds, (0, 255, 255))

        # Display the frame with cars spotted
        cv2.imshow("PRESS ESC TO QUIT", frame)

        # Wait for a keypress
        key = cv2.waitKey(1)

        # stop if "q" is pressed
        if key == 27:
            break

    print("Code Completed")


def loop(frame, type, color):

    import cv2

    # Draw rectangles around cars
    for (x, y, w, h) in type:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


if __name__ == "__main__":
    main()
