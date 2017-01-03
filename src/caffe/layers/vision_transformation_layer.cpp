#include <vector>
#include <omp.h>
#include <cmath>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/yodi_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	//==================================================================

	template<typename Dtype>
	void VisionTransformationLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);

        m_noiseStd = Dtype(0.1f);
        m_noiseMean = Dtype(0.0f);
        m_noiseStdSmall = Dtype(1.0f / 255.0f);
        m_rotateMinAngle = Dtype(-10.0f);
        m_rotateMaxAngle = Dtype(10.0f);
        m_rotateFillValue = Dtype(0.0f);
        m_interpolationMethod = cv::INTER_LINEAR;
        m_perPixelMultiplierMean = Dtype(1.0f);
        m_perPixelMultiplierStd = Dtype(0.0f);
        m_rescaleProbability = 0.0f;
        m_constantMultiplierMean = Dtype(1.0f);
        m_constantMultiplierStd = Dtype(0.0f);

        if (this->layer_param_.has_vision_transformation_param() == true)
        {
        	VisionTransformationParameter params = this->layer_param_.vision_transformation_param();
            m_noiseStd = params.noise_std();
    		m_noiseMean = params.noise_mean();
    		m_noiseStdSmall = params.noise_std_small();
    		m_rotateMinAngle = params.rotate_min_angle();
    		m_rotateMaxAngle = params.rotate_max_angle();
    		m_rotateFillValue = params.rotate_fill_value();
    		m_perPixelMultiplierMean = params.per_pixel_multiplier_mean();
    		m_perPixelMultiplierStd = params.per_pixel_multiplier_std();
    		m_rescaleProbability = params.rescale_probability();
    		m_constantMultiplierMean = params.constant_multiplier_mean();
    		m_constantMultiplierStd = params.constant_multiplier_std();
        }

        DCHECK(m_noiseStd >= 0);
        DCHECK(m_noiseStdSmall >= 0);
		DCHECK(m_rotateMinAngle < m_rotateMaxAngle);
	}

	//==================================================================

	template<typename Dtype>
	void VisionTransformationLayer<Dtype>::Reshape(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::Reshape(bottom, top);
	}

	//==================================================================

	template<typename Dtype>
	void VisionTransformationLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		Blob<Dtype>* bottomBlob = bottom[0];
		const Dtype* bottomData = bottomBlob->cpu_data();

		Blob<Dtype>* topBlob = top[0];
		Dtype* topData = topBlob->mutable_cpu_data();

		const int number = bottomBlob->num();
		const int width  = bottomBlob->width();
		const int height = bottomBlob->height();
		const int channels = bottomBlob->channels();

		cv::Size size(width, height);
		cv::Point2f center(width / 2.0, height / 2.0);

		//--------- angle distribution
		std::vector<float> angles(number, 0.0f);

		caffe_rng_uniform<float>(
				number,
				m_rotateMinAngle,
				m_rotateMaxAngle,
				angles.data());

		//--------- rescale distribution
		std::vector<int> rescales(number, 0);

		caffe_rng_bernoulli<float>(
				number,
				m_rescaleProbability,
				rescales.data());

		//--------- multiplier distribution
		std::vector<Dtype> multipliers(number, Dtype(0));

		if (m_constantMultiplierMean != 0.0f &&
			m_constantMultiplierStd != 0.0f)
		{
			caffe_rng_gaussian<Dtype>(
					number,
					m_constantMultiplierMean,
					m_constantMultiplierStd,
					multipliers.data());
		}

		//--------- iterate over images in blob
		#pragma omp parallel for
		for (int n = 0; n < number; n++)
		{
			float angle = angles[n];
			bool rescale = rescales[n] > 0;
			Dtype multiplier = std::max(Dtype(0.0), multipliers[n]);
			cv::Mat noiseSlice, noiseSmallSlice, perPixelMultiplierSlice;
			cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
			cv::Mat tmpSlice(height, width, cv::DataType<Dtype>::type);

			//----- add gaussian noise high
			if (m_noiseStd != 0.0f)
			{
				noiseSlice = cv::Mat(height, width, cv::DataType<Dtype>::type);
				caffe_rng_gaussian<Dtype>(
						width * height,
						Dtype(m_noiseMean),
						Dtype(m_noiseStd),
						(Dtype*)noiseSlice.data);
			}

			//----- add really small noise high
			if (m_noiseStdSmall != 0.0f)
			{
				noiseSmallSlice = cv::Mat(height, width, cv::DataType<Dtype>::type);
				caffe_rng_gaussian<Dtype>(
						width * height,
						Dtype(0),
						Dtype(m_noiseStdSmall),
						(Dtype*)noiseSmallSlice.data);
			}

		    //----- multiply pixel
		    if (m_perPixelMultiplierMean != Dtype(0.0) &&
		    	m_perPixelMultiplierStd != Dtype(0.0))
		    {
		    	perPixelMultiplierSlice = cv::Mat(height, width, cv::DataType<Dtype>::type);
				caffe_rng_gaussian<Dtype>(
						width * height,
						Dtype(m_perPixelMultiplierMean),
						Dtype(m_perPixelMultiplierStd),
						(Dtype*)perPixelMultiplierSlice.data);
		    }

			//----- iterate over channels
			for (int c = 0; c < channels; c++)
			{
				int offset = bottomBlob->offset(n, c, 0, 0);
				cv::Mat bottomSlice(height, width, cv::DataType<Dtype>::type, bottomData[offset]);

				//----- rotate
				if (angle == 0.0f)
				{
					bottomSlice.copyTo(tmpSlice);
				}
				else
				{
					cv::warpAffine(
							bottomSlice,
							tmpSlice,
							rotationMatrix,
							size,
							m_interpolationMethod,
							cv::BORDER_CONSTANT,
							Dtype(m_rotateFillValue));
				}

				//----- add gaussian noise high
				if (noiseSlice.empty() == false)
				{
					cv::add(tmpSlice, noiseSlice, tmpSlice);
				}

				//----- add sub pixel gaussian noise
				if (noiseSmallSlice.empty() == false)
				{
					cv::add(tmpSlice, noiseSmallSlice, tmpSlice);
				}

			    //----- multiply pixel
			    if (perPixelMultiplierSlice.empty() == false)
			    {
			    	cv::multiply(tmpSlice, perPixelMultiplierSlice, tmpSlice);
			    }

				//----- rescale
			    if (rescale == true)
			    {
			    	cv::Mat imageDown, imageDownResized;
			    	cv::pyrDown(tmpSlice, imageDown);
			    	cv::resize(
			    			imageDown,
							tmpSlice,
							cv::Size(width, height),
							0,
							0,
							m_interpolationMethod);
			    }

			    //----- multiply channel
			    if (multiplier != 0.0f &&
			    	multiplier != 1.0f)
			    {
			    	tmpSlice *= multiplier;
			    }

			    //---- copy to top blob
			    tmpSlice.copyTo(cv::Mat(height, width, cv::DataType<Dtype>::type, topData[offset]));
			}
		}
	}

	//==================================================================

	template<typename Dtype>
	void VisionTransformationLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype> *>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom)
	{
		if (propagate_down[0])
		{
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}

	//==================================================================

	#ifdef CPU_ONLY
	STUB_GPU(VisionTransformationLayer);
	#endif

	INSTANTIATE_CLASS(VisionTransformationLayer);
	REGISTER_LAYER_CLASS(VisionTransformation);
};
