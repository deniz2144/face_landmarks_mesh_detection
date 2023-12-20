import cv2
import mediapipe as mp

cap = cv2.VideoCapture("videos/video.mp4")
def face_mesh():
    while True:
        ret, frame = cap.read()
        # resize_img = cv2.resize(frame, (1920, 1080)) to resize what res you want to display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating facemesh variables
        mp_draw = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing_styles = mp.solutions.drawing_styles
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Facial landmarks
        result = face_mesh.process(rgb_image)

        for multi_face_landmark in result.multi_face_landmarks:
            # FACE TESSELATION
            mp_draw.draw_landmarks(frame,
                                   landmark_list=multi_face_landmark,
                                   connections=mp_face_mesh.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=False,
                                   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # FACE CONTOURS
            mp_draw.draw_landmarks(frame,
                                   landmark_list=multi_face_landmark,
                                   connections=mp_face_mesh.FACEMESH_CONTOURS,
                                   landmark_drawing_spec=False,
                                   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            # NOSE Detection
            mp_draw.draw_landmarks(frame,
                                   landmark_list=multi_face_landmark,
                                   connections=mp_face_mesh.FACEMESH_NOSE,
                                   landmark_drawing_spec=False)
            # IRISES Detection
            mp_draw.draw_landmarks(frame,
                               landmark_list=multi_face_landmark,
                               connections=mp_face_mesh.FACEMESH_IRISES,
                               landmark_drawing_spec=False,
                               connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())



            cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
        face_mesh()