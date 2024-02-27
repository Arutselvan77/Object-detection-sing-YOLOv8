import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

check_label = ["car", "person"]

# Get the frame width, height, and FPS of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_video.avi", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)[0]

        names = results.names
        # print("names  : " + str(names))
        labels = results.boxes.cls.tolist()
        # print("labels  : " + str(labels))
        coordinates = results.boxes.xyxy.tolist()

        try:
            unique_ids = results.boxes.id.tolist()
            # print("Unique IDsss   " + str(unique_ids))
        except:
            unique_ids = []

        # print("coordinates : " + str(coordinates))

        for coord, label, id in zip(coordinates, labels, unique_ids):
            if names[label] in check_label:

                # print(names[label])
                x1, y1, x2, y2 = map(int, coord)
                # cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
                # cv2.putText(img=frame, text=names[label], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.5, org=(x1, y1 - 10), color= (255, 255, 255), thickness=2)
                if len(unique_ids) != 0:
                    if names[label] == "car":
                        cv2.rectangle(
                            frame,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=(0, 0, 255),
                            thickness=2,
                        )
                        cv2.putText(
                            img=frame,
                            text="car " + str(int(id)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            org=(x1, y1 - 10),
                            color=(0, 0, 255),
                            thickness=2,
                        )
                    elif names[label] == "person":
                        cv2.rectangle(
                            frame,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=(0, 255, 0),
                            thickness=2,
                        )
                        cv2.putText(
                            img=frame,
                            text="person " + str(int(id)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            org=(x1, y1 - 10),
                            color=(255, 255, 255),
                            thickness=2,
                        )

        # coordinates = results.boxes.boxes.numpy().tolist()

        # print(coordinates)

        # Write the annotated frame to the output video
        out.write(frame)
        annotated_frame = results[0].plot()
        # print(str(annotated_frame) + "          dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        cv2.imshow("YOLOv8 Inference", frame)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
