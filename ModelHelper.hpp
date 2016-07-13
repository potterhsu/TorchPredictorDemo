//
// Created by Poter Hsu on 2016/2/26.
//

#ifndef TORCHPREDICTORDEMO_MODELHELPER_HPP
#define TORCHPREDICTORDEMO_MODELHELPER_HPP

#include <iostream>
#include <TorchPredictor/parser/BinaryModelParser.hpp>

using namespace std;
using namespace cv;

enum Mission {
    Face,
    Gender
};

template <typename DType>
class ModelHelper {
private:
    const double meanB = 129.1863;
    const double meanG = 104.7624;
    const double meanR = 93.5940;

    const char* faceCategories[4] {"Dami", "Darren", "Jones", "Poter"};
    const char* genderCategories[2] {"Male", "Female"};

public:
    const char** categories;
    Mission mission;

    ModelHelper(Mission mission_) : mission(mission_) {
        categories = mission == Mission::Face ? faceCategories : genderCategories;
    }

    void parseModel(const char* pathToModel, Module<DType>* &model) {
        BinaryModelParser binaryModelParser(pathToModel);
        model = binaryModelParser.parse<DType>();
        if (model == nullptr) {
            cerr << "Parse model failed" << endl;
            exit(-1);
        }
    }

    void composeInputFromImage(Mat img, shared_ptr<Tensor<DType>> input) {
        DType* pInput = input->data;
        const long inputStride0 = input->sizes[1] * input->sizes[2];

        for (int row = 0; row < img.rows; ++row) {
            for (int col = 0; col < img.cols; ++col) {
                uchar *pImageData = img.ptr(row, col);
                DType b = (DType) (pImageData[0] - meanB);
                DType g = (DType) (pImageData[1] - meanG);
                DType r = (DType) (pImageData[2] - meanR);
                pInput[0 * inputStride0] = r;
                pInput[1 * inputStride0] = g;
                pInput[2 * inputStride0] = b;
                ++pInput;
            }
        }
    }

    long calcMaxIndex(const shared_ptr<Tensor<DType>> tensor) {
        long maxIndex = 0;
        double max = -DBL_MAX;
        for (long i = 0; i < tensor->nElem; ++i) {
            DType value = tensor->data[i];
            if (value > max) maxIndex = i, max = value;
        }
        return maxIndex;
    }

    void printOutput(const shared_ptr<Tensor<DType>> output) {
        cout << "Output: " << endl;
        for (long i = 0; i < output->nElem; ++i) {
            cout << "  " << categories[i] << " = " << output->data[i] << endl;
        }
    }

    void printResult(const shared_ptr<Tensor<DType>> output) {
        long maxIndex = calcMaxIndex(output);
        cout << "Result is " << categories[maxIndex] << endl;
    }
};


#endif //TORCHPREDICTORDEMO_MODELHELPER_HPP

