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
	/* Example prototxt
	 *
		layer {
		  name: "data_vision"
		  type: "VisionTransformation"
		  bottom: "data"
		  top: "data"
		  vision_transformation_param {
			noise_std : 0.02
			noise_mean : 0.0
			noise_std_small : 0.0
			rotate_min_angle : -15.0
			rotate_max_angle : 15.0
			rotate_fill_value : 0.0
			per_pixel_multiplier_mean : 0.0
			per_pixel_multiplier_std : 0.0
			rescale_probability : 0.25
			constant_multiplier_mean : 1.0
			constant_multiplier_std : 0.1
			constant_multiplier_color_mean : 1.0
			constant_multiplier_color_std : 0.1
			scale_mean : 1.0
			scale_std : 0.1
			value_cap_min : -1
			value_cap_max : 1
		  }
		  include {
			phase: TRAIN
		  }
		}
	*/
	//==================================================================

	template<typename Dtype>
	void VisionTransformationLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);

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

    		m_scaleMean = params.scale_mean();
    		m_scaleStd = params.scale_std();
    		m_constantMultiplierColorMean = params.constant_multiplier_color_mean();
    		m_constantMultiplierColorStd = params.constant_multiplier_color_std();

    		m_valueCapMin = params.value_cap_min();
    		m_valueCapMax = params.value_cap_max();
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
		std::vector<Dtype> multipliers(number, Dtype(1.0));

		if (m_constantMultiplierMean != 0.0f &&
			m_constantMultiplierStd != 0.0f)
		{
			caffe_rng_gaussian<Dtype>(
					number,
					m_constantMultiplierMean,
					m_constantMultiplierStd,
					multipliers.data());
		}

		//--------- color multiplier distribution
		std::vector<Dtype> multipliersColor(number * channels, Dtype(1.0));

		if (m_constantMultiplierColorMean != 0.0f &&
			m_constantMultiplierColorStd != 0.0f)
		{
			caffe_rng_gaussian<Dtype>(
					number * channels,
					m_constantMultiplierColorMean,
					m_constantMultiplierColorStd,
					multipliersColor.data());
		}

		//--------- scale distribution
		std::vector<float> scales(number, 1.0f);

		if (m_scaleMean != 0.0f &&
			m_scaleStd != 0.0f)
		{
			caffe_rng_gaussian<float>(
					number,
					m_scaleMean,
					m_scaleStd,
					scales.data());
		}

		//--------- iterate over images in blob
		#pragma omp parallel for
		for (int n = 0; n < number; n++)
		{
			float angle = angles[n];
			float scale = scales[n];
			bool rescale = rescales[n] > 0;
			Dtype multiplier = std::max(Dtype(0.0), multipliers[n]);

			cv::Mat noiseSlice, noiseSmallSlice, perPixelMultiplierSlice;
			cv::Mat rotationMatrix =
					cv::getRotationMatrix2D(
							center,
							double(angle),
							double(scale));
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
				Dtype multiplierColor = multipliersColor[n * channels + c];
				Dtype multiplierTotal = multiplier * multiplierColor;
				int offset = bottomBlob->offset(n, c, 0, 0);
				cv::Mat bottomSlice(height, width, cv::DataType<Dtype>::type, bottomData[offset]);

				//----- rotate
				if (angle == 0.0f && scale == 0.0f)
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
			    if (multiplierTotal != Dtype(0.0) &&
			    	multiplierTotal != Dtype(1.0))
			    {
			    	tmpSlice *= multiplierTotal;
			    }

			    //----- cap everything between min and max
			    if (m_valueCapMin != m_valueCapMax)
			    {
			    	for (cv::MatIterator_<Dtype> iter = tmpSlice.begin<Dtype>();
			    			iter != tmpSlice.end<Dtype>();
			    			++iter)
			    	{
			    		Dtype tmp = *iter;
			    		tmp = std::max(m_valueCapMin, std::min(tmp, m_valueCapMax));
			    		*iter = tmp;
			    	}
			    }

			    //---- copy to top blob
			    tmpSlice.copyTo(
			    		cv::Mat(
			    			height,
			    			width,
			    			cv::DataType<Dtype>::type,
							topData[offset]));
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
	
	INSTANTIATE_CLASS(VisionTransformationLayer);
	REGISTER_LAYER_CLASS(VisionTransformation);
};
