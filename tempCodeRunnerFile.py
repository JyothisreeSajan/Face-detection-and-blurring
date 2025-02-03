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