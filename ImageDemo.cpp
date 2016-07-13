#include <iostream>
#include <opencv2/opencv.hpp>
#include "ModelHelper.hpp"

using namespace std;
using namespace cv;

typedef float DType;

const int FaceWidth = 224;
const int FaceHeight = 224;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Wrong argument" << endl;
        cerr << "Usage: ImageDemo [Mission] /path/to/Model.tpb /path/to/Image.jpg" << endl;
        cerr << "   Mission: 1 = Face, 2 = Gender" << endl;
        exit(-1);
    }

    ModelHelper<DType> modelHelper(*argv[1] - 0x30 == 1 ? Mission::Face : Mission::Gender);
    Module<DType>* model;

    const char* pathToModel = argv[2];
    const char* pathToImage = argv[3];

    cout << "Parsing model..." << endl;
    modelHelper.parseModel(pathToModel, model);
    shared_ptr<Tensor<DType>> input = make_shared<Tensor<DType>>(vector<long>({3, FaceHeight, FaceWidth}));

    cout << "Load image to input..." << endl;
    Mat img = imread(pathToImage);
    resize(img, img, Size(FaceWidth, FaceHeight));
    modelHelper.composeInputFromImage(img, input);

    cout << "Starting forward..." << endl;
    shared_ptr<Tensor<DType>> output = model->forward(input);

    modelHelper.printOutput(output);
    modelHelper.printResult(output);

    delete model;
    return 0;
}
