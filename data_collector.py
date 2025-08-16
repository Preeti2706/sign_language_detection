import os
import cv2
import time
import uuid
import matplotlib.pyplot as plt

IMAGE_PATH = "CollectedImages"

labels = ['Hello', 'Yes', 'No', 'Thanks', 'IloveYou', 'Please']
number_of_images = 5

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)  # avoid error if folder already exists

    # open camera
    cap = cv2.VideoCapture(0)
    print(f"\n📸 Collecting images for: {label}")
    time.sleep(3)  # pause before capturing

    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame, skipping...")
            continue

        # build filename
        imagename = os.path.join(img_path, f"{label}.{str(uuid.uuid1())}.jpg")

        # save image
        cv2.imwrite(imagename, frame)
        print(f"✅ Saved image {imagename}")

        # preview using matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"{label} - {imgnum+1}/{number_of_images}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1)  # show for 1 sec
        plt.close()

        time.sleep(1)  # delay before next capture

    cap.release()

print("\n🎉 Image collection complete!")