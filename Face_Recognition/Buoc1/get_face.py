import argparse
import numpy as np
import cv2 as cv

# Convert string arguments to boolean
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

# Parse command-line arguments
parser = argparse.ArgumentParser()
""" parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.') """
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor for resizing frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='../model/face_detection_yunet_2023mar.onnx', help='Path to the face detection model.')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='../model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Threshold for filtering faces.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Non-Maximum Suppression threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Top-K bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default="yes", help='Flag to save results.')
args = parser.parse_args()

# Function to visualize detection results
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
            for i in range(4, 13, 2):
                cv.circle(input, (coords[i], coords[i + 1]), 2, (255, 0, 0), thickness)
    cv.putText(input, f'FPS: {fps:.2f}', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Main execution
if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model, "", (320, 320), args.score_threshold, args.nms_threshold, args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model, "")

    tm = cv.TickMeter()
    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        key = cv.waitKey(1)
        if key == 27:
            break

        if key in [ord('s'), ord('S')] and faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            file_name = f'../image/HoangAnh/HoangAnh_{dem:04d}.bmp'
            cv.imwrite(file_name, face_align)
            dem += 1

        visualize(frame, faces, tm.getFPS())
        cv.imshow('Live', frame)

    cv.destroyAllWindows()
