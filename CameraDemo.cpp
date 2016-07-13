#include <iostream>
#include <opencv2/opencv.hpp>
#include "ModelHelper.hpp"

using namespace std;
using namespace cv;

typedef float DType;

const int FaceWidth = 224;
const int FaceHeight = 224;

Module<DType>* model;
ModelHelper<DType>* modelHelper;
shared_ptr<Tensor<DType>> input;

CascadeClassifier faceCascade;
char fps[BUFSIZ];

void detectAndDisplay(Mat frame) {
    double t = getTickCount();
    std::vector<Rect> faces;
    Mat frameGray;

    cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    equalizeHist(frameGray, frameGray);

    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0, Size(80, 80));

    for(size_t i = 0; i < faces.size(); ++i) {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        Rect roi(
            Point(int(center.x - faces[i].width / 1.5), int(center.y - faces[i].height / 1.5)),
            Point(int(center.x + faces[i].width / 1.5), int(center.y + faces[i].height / 1.5))
        );

        // Ignore frame if ROI out of range
        if (roi.tl().x < 0 || roi.tl().y < 0 ||
                roi.br().x >= frame.size().width || roi.br().y >= frame.size().height)
            break;

        rectangle(frame, roi, Scalar(255, 0, 0));
//        rectangle(frame, faces[i], Scalar(0, 255, 0));

        Mat face = frame(roi);
        resize(face, face, Size(FaceWidth, FaceHeight));

        modelHelper->composeInputFromImage(face, input);
        shared_ptr<Tensor<DType>> output = model->forward(input);
        long maxIndex = modelHelper->calcMaxIndex(output);
        const char* category = modelHelper->categories[maxIndex];

        putText(frame, category, Point(roi.x, roi.y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));

        // Process only first face
        break;
    }

    t = (getTickCount() - t) / getTickFrequency();
    sprintf(fps, "FPS: %lf", 1.0 / t);
    putText(frame, fps, Point(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));

    imshow("Demo", frame);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Wrong argument" << endl;
        cerr << "Usage: CameraDemo [Mission] /path/to/Model.tpb /path/to/cascade_face.xml" << endl;
        cerr << "   Mission: 1 = Face, 2 = Gender" << endl;
        exit(-1);
    }

    modelHelper = new ModelHelper<DType>(*argv[1] - 0x30 == 1 ? Mission::Face : Mission::Gender);

    const char* pathToModel = argv[2];

    cout << "Parsing model..." << endl;
    modelHelper->parseModel(pathToModel, model);

    cout << "Ready" << endl;

    input = make_shared<Tensor<DType>>(vector<long>({3, FaceHeight, FaceWidth}));

    VideoCapture capture;
    Mat frame;

    const char* pathToCascadeFace = argv[3];
    if(!faceCascade.load(pathToCascadeFace)) {
        cerr << "Load face cascade failed" << endl;
        return -1;
    };

    capture.open(-1);

    if (!capture.isOpened()) {
        cerr << "Opening video capture failed" << endl;
        return -1;
    }

    while (capture.read(frame)) {
        if(frame.empty()) {
            cout << " No captured frame -- Break" << endl;
            break;
        }

        detectAndDisplay(frame);

        int c = waitKey(10);
        if((char)c == 27) break;
    }

    delete model;
    delete modelHelper;
    return 0;
}
