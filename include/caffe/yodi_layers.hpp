#ifndef CAFFE_YODI_LAYERS_HPP_
#define CAFFE_YODI_LAYERS_HPP_

#include <string>
#include <vector>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

namespace caffe
{
	//==================================================================
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

    //==================================================================

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

    //==================================================================

    /*
     * Splitter Layer
     */
    template<typename Dtype>
    class VisionTransformationLayer :
    		public NeuronLayer<Dtype>
    {
        public:
    		//----------------------------------
            explicit
			VisionTransformationLayer(const LayerParameter &param)
            	: NeuronLayer< Dtype>(param)
			{
                m_noiseStd = Dtype(0.1f);
                m_noiseMean = Dtype(0.0f);
                m_noiseStdSmall = Dtype(1.0f / 255.0f);
                m_rotateMinAngle = Dtype(-10.0f);
                m_rotateMaxAngle = Dtype(10.0f);
                m_rotateFillValue = Dtype(0.0f);
#ifdef USE_OPENCV
                m_interpolationMethod = cv::INTER_LINEAR;
#else
                m_interpolationMethod = 0;
#endif
                m_perPixelMultiplierMean = Dtype(1.0f);
                m_perPixelMultiplierStd = Dtype(0.0f);
                m_rescaleProbability = 0.0f;
                m_constantMultiplierMean = Dtype(1.0f);
                m_constantMultiplierStd = Dtype(0.0f);
                m_scaleMean = 1.0f;
                m_scaleStd = 0.0f;
                m_constantMultiplierColorMean = Dtype(1);
                m_constantMultiplierColorStd = Dtype(0);
                m_valueCapMin = Dtype(0);
                m_valueCapMax = Dtype(0);
                m_maxoutPassthroughProbabilityIteration = 0;
                m_passthroughProbability = 1.0f;
			}
            //----------------------------------
            virtual ~VisionTransformationLayer(){}
            //----------------------------------
            virtual void LayerSetUp(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            //----------------------------------
            virtual void Reshape(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            //----------------------------------
            virtual inline const char* type() const { return "VisionTransformation"; }
            //----------------------------------
            virtual inline int MinTopBlobs() const { return 1; }
            //----------------------------------
            virtual inline int MaxTopBlobs() const { return 1; }
            //----------------------------------
        protected:
            //----------------------------------
            virtual void Forward_cpu(
            		const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
            //----------------------------------
            virtual void Backward_cpu(
            		const vector<Blob < Dtype> *> &top,
					const vector<bool> &propagate_down,
					const vector<Blob < Dtype> *> &bottom);
            //----------------------------------
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
            //---- Passthrough probability
            // Indicates the probability
            // of just pushing the image without transformations
            float m_passthroughProbability;
            // indicates at which iteration the passthrough probabilit
            // reaches its high
            int m_maxoutPassthroughProbabilityIteration;
    };

    //==================================================================

    /**
     * @brief Performs a histogram reduction in a sliding window fashion,
     *   creating a histogram for each window
     *
     */
    template <typename Dtype>
    class SlidingHistogramLayer :
    		public Layer<Dtype>
    {
    	protected:
    		//----------------------------------
    		virtual void
			Forward_cpu(
					const vector<Blob<Dtype>*>& bottom,
    				const vector<Blob<Dtype>*>& top);
    		//----------------------------------
    		virtual void
			Backward_cpu(
					const vector<Blob<Dtype>*>& top,
					const vector<bool>& propagate_down,
					const vector<Blob<Dtype>*>& bottom);
    		//----------------------------------
    		Dtype m_min;
    		Dtype m_max;
    		int m_bins;
    		int m_kernel_h_, m_kernel_w_;
    		int m_stride_h_, m_stride_w_;
    		int m_channels_;
    		int m_top_channels_;
    		int m_height_, m_width_;
    		int m_result_height_, m_result_width_;
    		Dtype m_diffuseValue;
    		Dtype m_multiplier;
    		bool m_flatten;
    		// holds indices of mapping bottom value to top bin
    		Blob<int> m_idx_;
    		Blob<int> m_distance;
    		//----------------------------------
    	public:
    		explicit SlidingHistogramLayer(const LayerParameter& param)
    			: Layer<Dtype>(param)
			{
    			m_bins = 0;
    			m_min = Dtype(0);
    			m_max = Dtype(0);
    			m_kernel_h_ = 0;
    			m_kernel_w_ = 0;
    			m_stride_h_ = 0;
    			m_stride_w_ = 0;
    			m_height_ = 0;
    			m_width_ = 0;
    			m_channels_ = 0;
    			m_top_channels_ = 0;
    			m_result_height_ = 0;
    			m_result_width_ = 0;
    			m_diffuseValue = Dtype(0);
    			m_multiplier = Dtype(0);
    			m_flatten = false;
			}
    		//----------------------------------
    		virtual ~SlidingHistogramLayer(){}
    		//----------------------------------
    		virtual void
			LayerSetUp(
				const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
    		//----------------------------------
    		virtual void Reshape(
    			const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);

    		//----------------------------------
    		virtual inline const char*
			type() const { return "SlidingHistogram"; }
    		//----------------------------------
    		virtual inline int ExactNumBottomBlobs() const { return 1; }
    		//----------------------------------
    		virtual inline int MinTopBlobs() const { return 1; }
    		//----------------------------------
    		virtual inline int MaxTopBlobs() const { return 1; }
    		//----------------------------------
    		virtual inline int
			bin_of_value(const Dtype value) const
    		{
    			if (value >= m_max)
    			{
    				return m_bins-1;
    			}

    			if (value <= m_min)
    			{
    				return 0;
    			}

    			return int(floor(
    					(value - m_min) * m_multiplier));
    		}
    		//----------------------------------
    };

    //==================================================================

    /**
     * @brief Quantizes the input
     *
     */
    template <typename Dtype>
    class QuantizeLayer :
    		public NeuronLayer<Dtype>
    {
    	protected:
    		//----------------------------------
    		virtual void
			Forward_cpu(
					const vector<Blob<Dtype>*>& bottom,
    				const vector<Blob<Dtype>*>& top);
    		//----------------------------------
    		virtual void
			Backward_cpu(
					const vector<Blob<Dtype>*>& top,
					const vector<bool>& propagate_down,
					const vector<Blob<Dtype>*>& bottom);
    		//----------------------------------
    		Dtype m_min;
    		Dtype m_max;
    		Dtype m_div;
    		int m_bins;
    		//----------------------------------
    	public:
    		explicit QuantizeLayer(const LayerParameter& param)
    			: NeuronLayer<Dtype>(param)
			{
    			m_bins = 8;
    			m_min = Dtype(-1);
    			m_max = Dtype(+1);
			}
    		//----------------------------------
    		virtual ~QuantizeLayer(){}
    		//----------------------------------
    		virtual void
			LayerSetUp(
				const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
    		//----------------------------------
    		virtual void Reshape(
    			const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
    		//----------------------------------
    		virtual inline const char*
			type() const { return "Quantize"; }
    		//----------------------------------
    		virtual inline int ExactNumBottomBlobs() const { return 1; }
    		//----------------------------------
    		virtual inline int MinTopBlobs() const { return 1; }
    		//----------------------------------
    		virtual inline int MaxTopBlobs() const { return 1; }
    		//----------------------------------
    };

	//=========================================================

	template<typename Dtype>
	class DataDrivenDropoutLayer :
			public NeuronLayer<Dtype>
	{
		protected:
			//-----------------------------------------
			virtual void
			Forward_cpu(
					const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
			//-----------------------------------------
			virtual void
			Backward_cpu(
					const vector<Blob < Dtype> *> &top,
					const vector<bool> &propagate_down,
					const vector<Blob < Dtype> *> &bottom);
			//-----------------------------------------
			// percentage to keep [0, 1]
			float m_topk;
			// if true work on channels else work on weights
			DataDrivenDropoutParameter_FilterMethod m_filter_method;
			// probability of passthrough the data
			float m_passthrough_probability;
			// true if output is passed, false otherwise
			Blob<int> m_mask;
			// adjust scale based on dropout effect
			Dtype m_scale;
			//-----------------------------------------
    	public:
			//-----------------------------------------
			explicit DataDrivenDropoutLayer(const LayerParameter &param)
					: NeuronLayer<Dtype>(param)
			{
				m_topk = 1.0;
				m_scale = Dtype(1);
				m_passthrough_probability = 0.0f;
				m_filter_method = DataDrivenDropoutParameter_FilterMethod::DataDrivenDropoutParameter_FilterMethod_FULL;
			}
			//-----------------------------------------
			virtual void
			LayerSetUp(const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
			//-----------------------------------------
			virtual void
			Reshape(const vector<Blob < Dtype> *> &bottom,
					const vector<Blob < Dtype> *> &top);
			//-----------------------------------------
			virtual inline const char *type() const
			{ return "DataDrivenDropout"; }
			//-----------------------------------------
			virtual inline int ExactNumBottomBlobs() const
			{ return 1; }
			//-----------------------------------------
			virtual inline int MinTopBlobs() const
			{ return 1; }
			//-----------------------------------------
			virtual inline int MaxTopBlobs() const
			{ return 1; }
			//-----------------------------------------
	};

    //==================================================================
}  // namespace caffe

#endif  // CAFFE_YODI_LAYERS_HPP_
