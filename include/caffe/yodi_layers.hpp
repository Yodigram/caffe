#ifndef CAFFE_YODI_LAYERS_HPP_
#define CAFFE_YODI_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

namespace caffe {
    /*
     * Warp Layer
     */
    template<typename Dtype>
    class UnwarpLayer :
    		public Layer<Dtype>
    {
        public:
            explicit UnwarpLayer(const LayerParameter &param)
                    : Layer<Dtype>(param) { }
            virtual void LayerSetUp(const vector<Blob < Dtype> *> &bottom, const vector<Blob < Dtype> *> &top);
            virtual void Reshape(const vector<Blob < Dtype> *> &bottom, const vector<Blob < Dtype> *> &top);

            virtual inline const char *type() const { return "Unwarp"; }
            virtual inline int ExactNumBottomBlobs() const { return 2; }
            virtual inline int MinTopBlobs() const { return 1; }
            virtual inline int MaxTopBlobs() const { return 3; }

        protected:
            virtual void Forward_cpu(const vector<Blob < Dtype> *> &bottom,
                                     const vector<Blob < Dtype> *> &top);
            virtual void Backward_cpu(const vector<Blob < Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob < Dtype> *> &bottom);

            float getTH(int h, int w);
            Dtype _defaultValue;
            Dtype _maxStep;
            int   _noBlocks;
            int   _blockSize;

            #ifdef USE_OPENCV
            cv::Mat  cv_img_wrap;
            cv::Mat  cv_img_unwrap;
            cv::Mat  cv_img_dist_h;
            cv::Mat  cv_img_dist_w;
            cv::Mat  cv_img_map;
            #endif
    };


    /////////////////////////////////////////////////////////////////////////

    /*
     * Splitter Layer
     */
    template<typename Dtype>
    class SplitterLayer :
    		public Layer<Dtype>
    {
        public:
            explicit SplitterLayer(const LayerParameter &param)
                    : Layer<Dtype>(param) { }

            virtual void LayerSetUp(const vector<Blob < Dtype> *> &bottom,
                                    const vector<Blob < Dtype> *> &top);
            virtual void Reshape(const vector<Blob < Dtype> *> &bottom,
                                 const vector<Blob < Dtype> *> &top);
            virtual inline const char *type() const { return "Splitter"; }
            virtual inline int ExactNumBottomBlobs() const { return 1; }
            virtual inline int MinTopBlobs() const { return 2; }
            virtual inline int MaxTopBlobs() const { return 10; }

        protected:
            virtual void Forward_cpu(const vector<Blob < Dtype> *
            > &bottom,
            const vector<Blob < Dtype> *> &top);
            virtual void Backward_cpu(const vector<Blob < Dtype> *
            > &top,
            const vector<bool> &propagate_down,
            const vector<Blob < Dtype> *> &bottom);
    };


    /////////////////////////////////////////////////////////////////////////

    /*
     * Splitter Layer
     */
    template<typename Dtype>
    class VisionTransformationLayer :
    		public NeuronLayer<Dtype>
    {
        public:
            explicit VisionTransformationLayer(const LayerParameter &param)
                    : NeuronLayer< Dtype>(param) { }

            virtual void LayerSetUp(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            virtual void Reshape(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            virtual inline const char *type() const { return "VisionTransformation"; }

            virtual inline int MinTopBlobs() const { return 1; }
            virtual inline int MaxTopBlobs() const { return 1; }

        protected:
            virtual void Forward_cpu(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            virtual void Backward_cpu(
            		const vector<Blob < Dtype> *> &top,
					const vector<bool> &propagate_down,
					const vector<Blob < Dtype> *> &bottom);
            //---- noise in the scale of the image
            float m_noiseStd;
            float m_noiseMean;
            //---- sub pixel noise
            float m_noiseStdSmall;
            //---- rotation min and max angle
            float m_rotateMinAngle;
            float m_rotateMaxAngle;
            Dtype m_rotateFillValue;
            //---- resizing
            float m_scaleMean;
            float m_scaleStd;
            float m_rescaleProbability;
            //---- Constant multiplier (whole image)
            Dtype m_constantMultiplierMean;
            Dtype m_constantMultiplierStd;
            //---- per pixel multiplier
            Dtype m_perPixelMultiplierMean;
            Dtype m_perPixelMultiplierStd;
            //---- interpolation method
            int m_interpolationMethod;
            //---- Color space shifting
            Dtype m_constantMultiplierColorMean;
            Dtype m_constantMultiplierColorStd;
            //---- Value min max cap
            Dtype m_valueCapMin;
            Dtype m_valueCapMax;
    };
}  // namespace caffe

#endif  // CAFFE_YODI_LAYERS_HPP_
