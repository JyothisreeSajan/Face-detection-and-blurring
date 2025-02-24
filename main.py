import os
import cv2
import argparse
import mediapipe as mp

def process_img(img, face_detection):
    if img is None:
        return None
    H, W, _= img.shape
    img_rgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1,y1,w,h = bbox.xmin , bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

        #blur faces
            img[y1: y1 + h , x1:x1 + w, :]= cv2.blur(img[y1: y1 + h , x1: x1 + w, :],(10 ,10))

    return img
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='webcam')
    parser.add_argument("--filePath", default=None)
    args = parser.parse_args()

    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #detect images
    mp_face_detection  = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
        if args.mode in ["image"]:
            #read image 
            img =cv2.imread(args.filePath)
            if img is None:
                print(f"Error: Could not read image file: {args.filePath}")
                return
            img = process_img(img, face_detection)

            if img is not None:
                cv2.imwrite(os.path.join(output_dir,"output.png"), img)
        
        elif args.mode == 'video':
            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                print("Error: Cannot open video file.")
            
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read video.")
            
            output_video = cv2.VideoWriter(os.path.join(output_dir,'output.mp4'),
                                        cv2.VideoWriter.fourcc(*'avc1'),
                                        25,
                                        (frame.shape[1], frame.shape[0])
                                        )
            
            while ret:
                    processed_frame = process_img(frame, face_detection)
                    if processed_frame is not None:
                        output_video.write(processed_frame)
                    ret, frame = cap.read()
                
                # Release resources
            cap.release()
            output_video.release()
        elif args.mode.lower() == 'webcam':  # Made case-insensitive
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot open webcam.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read from webcam.")
                    break

                processed_frame = process_img(frame, face_detection)
                if processed_frame is not None:
                    cv2.imshow("Webcam Face Blur", processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()            



        
                