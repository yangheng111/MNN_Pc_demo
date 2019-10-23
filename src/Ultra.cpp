#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <time.h>
#include "Ultra.h"
#define SCORE_NAME "scores"
#define BOX_NAME "460"


using namespace MNN;
using namespace MNN::CV;


typedef struct Bbox
{
    int x;
    int y;
    int w;
    int h;
    float score;

}Bbox;

bool sort_score(Bbox box1,Bbox box2)
{
    return (box1.score > box2.score);

}

float iou(Bbox box1,Bbox box2)
{
    int x1 = std::max(box1.x,box2.x);
    int y1 = std::max(box1.y,box2.y);
    int x2 = std::min((box1.x + box1.w),(box2.x + box2.w));
    int y2 = std::min((box1.y + box1.h),(box2.y + box2.h));
    float over_area = (x2 - x1) * (y2 - y1);
    float iou = over_area/(box1.w * box1.h + box2.w * box2.h-over_area);
    return iou;
}
//方法2  这种执行效率更高
std::vector<Bbox> nms(std::vector<Bbox>&vec_boxs,float threshold)
{
    std::vector<Bbox>results;
    std::vector<int> erase_index;
    std::sort(vec_boxs.begin(),vec_boxs.end(),sort_score);
    for(int i =0;i <vec_boxs.size()-1;i++)
    {
        for(int j =i+1;j <vec_boxs.size();j++)
        {
            if (std::find(erase_index.begin(), erase_index.end(), j)==erase_index.end())
            {
                float iou_value =iou(vec_boxs[i],vec_boxs[j]);
                if (iou_value >threshold)
                {
                    erase_index.push_back(j);
                }
            }
            
        }
    }
    for(int i =0;i <vec_boxs.size();i++)
    {   
        if (std::find(erase_index.begin(), erase_index.end(), i)==erase_index.end())
        {
            results.push_back(vec_boxs[i]);
        }
    }
    return results;
}




int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./pictureDetect.out model.mnn input.jpg out.jpg");
        return 0;
    }

    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));

    ScheduleConfig config;
    config.numThread = 1;
    config.type  = MNN_FORWARD_AUTO;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 4;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto output_scores = net->getSessionOutput(session, SCORE_NAME);
    auto output_boxes = net->getSessionOutput(session, BOX_NAME);

    auto dims    = input->shape();
    int inputDim = 0;
    int size_w   = 0;
    int size_h   = 0;
    int bpp      = 0;
    bpp          = input->channel();
    size_h       = input->height();
    size_w       = input->width();
    if (bpp == 0)
        bpp = 1;
    if (size_h == 0)
        size_h = 1;
    if (size_w == 0)
        size_w = 1;
    MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

    auto inputPatch = argv[2];
    int width, height, channel;
    auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPatch);
        return 0;
    }
    MNN_PRINT("origin size: %d, %d\n", width, height);

    clock_t start = clock();

    const float means[3] = {127.0f, 127.0f, 127.0f};
    const float norms[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
    CV::ImageProcess::Config preProcessConfig;
    ::memcpy(preProcessConfig.mean, means, sizeof(means));
    ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
    preProcessConfig.sourceFormat = CV::RGBA;
    preProcessConfig.destFormat   = CV::RGB;
    preProcessConfig.filterType   = CV::BILINEAR;

    auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
    CV::Matrix trans;
    // Set transform, from dst scale to src, the ways below are both ok
    trans.postScale(1.0 / size_w, 1.0 / size_h);
    trans.postScale(width, height);
    pretreat->setMatrix(trans);
    const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);
    pretreat->convert(rgbaPtr, width, height, 0, input);

    clock_t end0 = clock();
    float duration0 = (float) (end0 - start) / CLOCKS_PER_SEC;
    MNN_PRINT("transform time: %f\n", duration0);
    //save image
//        auto values = input->host<float>();
//        std::shared_ptr<Tensor> outputUser(new Tensor(input, Tensor::TENSORFLOW));
//        MNN_PRINT("output size:%d x %d x %d\n", outputUser->width(), outputUser->height(), outputUser->channel());
//        input->copyToHostTensor(outputUser.get());
//
//        auto swidth = outputUser->width();
//        auto sheight = outputUser->height();
//        auto schannel = outputUser->channel();
//        std::shared_ptr<Tensor> wrapTensor(CV::ImageProcess::createImageTensor<uint8_t>(outputUser->width(), outputUser->height(), 3, nullptr));
//        for (int y = 0; y < sheight; ++y) {
//            auto rgbaY = wrapTensor->host<uint8_t>() + 3 * y * swidth;
//            auto sourceY = outputUser->host<float>() + y * swidth * schannel;
//            for (int x=0; x<swidth; ++x) {
//                auto sourceX = sourceY + schannel * x;
//                int index = 0;
//                float maxValue = sourceX[0];
//                auto rgba = rgbaY + 3 * x;
//                for (int c=1; c<channel; ++c) {
//                    rgba[c] = sourceX[c];
//                }
//
//            }
//        }
//        stbi_write_png("input.png", size_w, size_h, 3, wrapTensor->host<uint8_t>(), size_w * 3);
//    }


    // run...
    net->runSession(session);

    clock_t end1 = clock();
    float duration1 = (float) (end1 - start) / CLOCKS_PER_SEC;
    MNN_PRINT("run time: %f\n", duration1);

    auto scoresdimType = output_scores->getDimensionType();
    if (output_scores->getType().code != halide_type_float) {
        scoresdimType = Tensor::TENSORFLOW;
    }
    std::shared_ptr<Tensor> outputScoresUser(new Tensor(output_scores, scoresdimType));
    MNN_PRINT("output Scores size:%d\n", outputScoresUser->elementSize());

    output_scores->copyToHostTensor(outputScoresUser.get());
    // auto ScoresType = outputScoresUser->getType();
    // auto ScoresSize = outputScoresUser->elementSize();

    // if (ScoresType.code == halide_type_float) {
        // auto ScoresValues = outputScoresUser->host<float>();
    //     for (int i = 0; i < ScoresSize; i = i+2) {
    //         if (ScoresValues[i+1] > 0.5){
    //         MNN_PRINT("score float noface:%f,face:%f\n", ScoresValues[i],ScoresValues[i+1]);
    //         }
    //     }
    // }
    // if (ScoresType.code == halide_type_uint && ScoresType.bytes() == 1) {
    //     auto ScoresValues = outputScoresUser->host<uint8_t>();
    //     for (int i = 0; i < ScoresSize; ++i) {
    //         if (ScoresValues[i+1] > 0.5){
    //         MNN_PRINT("score unint8 noface:%f,face:%f\n", ScoresValues[i],ScoresValues[i+1]);
    //         }
    //     }
    // }
    


    auto boxes_dimType = output_boxes->getDimensionType();
    if (output_boxes->getType().code != halide_type_float) {
        boxes_dimType = Tensor::TENSORFLOW;
    }
    std::shared_ptr<Tensor> outputBoxesUser(new Tensor(output_boxes, boxes_dimType));
    MNN_PRINT("output boxes size:%d\n", outputBoxesUser->elementSize());

    output_boxes->copyToHostTensor(outputBoxesUser.get());
    // auto BoxesType = outputBoxesUser->getType();
    // auto BoxesSize = outputBoxesUser->elementSize();

    // if (BoxesType.code == halide_type_float) {
    //     auto BoxesValues = outputBoxesUser->host<float>();
    //     for (int i = 0; i < BoxesSize; i = i+4) {
    //         MNN_PRINT(" face float cor:%f ,%f ,%f ,%f\n", BoxesValues[i],BoxesValues[i+1],BoxesValues[i+2],BoxesValues[i+3]);
    //     }
    // }
    // if (BoxesType.code == halide_type_uint && BoxesType.bytes() == 1) {
    //     auto BoxesValues = outputBoxesUser->host<uint8_t>();
    //     for (int i = 0; i < BoxesSize; i=i+4) {
    //         MNN_PRINT(" face unint8 cor:%f ,%f ,%f ,%f\n", BoxesValues[i],BoxesValues[i+1],BoxesValues[i+2],BoxesValues[i+3]);
    //     }
    // }

    auto ScoresValues = outputScoresUser->host<float>();
    auto BoxesValues = outputBoxesUser->host<float>();
    std::vector<Bbox> vec_boxs;
    for(int i = 0; i < OUTPUT_NUM; ++i)
    {   

        float xcenter =     BoxesValues[i*4 + 0 ] * center_variance * anchors[2][i] + anchors[0][i];
        float ycenter =     BoxesValues[i*4 + 1 ] * center_variance * anchors[3][i] + anchors[1][i];
        float w       = exp(BoxesValues[i*4 + 2 ] * size_variance) * anchors[2][i];
        float h       = exp(BoxesValues[i*4 + 3 ] * size_variance) * anchors[3][i];

        float ymin    = ( ycenter - h * 0.5 ) * height;
        float xmin    = ( xcenter - w * 0.5 ) * width;
        float ymax    = ( ycenter + h * 0.5 ) * height;
        float xmax    = ( xcenter + w * 0.5 ) * width;

        // probability decoding, softmax
        float nonface_prob = ScoresValues[i*2 + 0];
        float face_prob    = ScoresValues[i*2 + 1];
        if ( face_prob > face_prob_thresh )
        {
            int int_x = xmin;
            int int_y = ymin;
            int int_width = xmax - xmin;
            int int_height = ymax - ymin;
            float score = face_prob;
            Bbox item;
            item.x = int_x;
            item.y = int_y;
            item.w = int_width;
            item.h = int_height;
            item.score = score;
            vec_boxs.push_back(item);
        }
    }
    // for(int i =0; i<vec_boxs.size();i++)
    // {
    //     std::cout<<vec_boxs[i].score<<"___"<<vec_boxs[i].x << "___"<<vec_boxs[i].y<<"___"<<vec_boxs[i].w<<"___"<<vec_boxs[i].h<<std::endl;
    // }

    std::vector<Bbox> vec_results=nms(vec_boxs,nms_thresh);
    for(int i =0; i<vec_results.size();i++)
    {
        std::cout<<vec_results[i].score<<"___"<<vec_results[i].x << "___"<<vec_results[i].y<<"___"<<vec_results[i].w<<"___"<<vec_results[i].h<<std::endl;
    }
    clock_t end = clock();
    float duration = (float) (end - start) / CLOCKS_PER_SEC;
    MNN_PRINT("time: %f\n", duration);

    return 0;
}
