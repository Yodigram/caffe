#include <vector>
#include <omp.h>
#include <cmath>
#include <caffe/caffe.hpp>
#include <opencv2/gpu/gpu.hpp>
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
			passthrough_probability : 0.5
			maxout_passthrough_probability_iteration : 100000
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
	    CHECK(this->layer_param_.has_vision_transformation_param() == true)
			<< "Vision transformation parameters are missing";

		VisionTransformationParameter params = this->layer_param_.vision_transformation_param();

		if (params.has_noise_std())
		{
			m_noiseStd = params.noise_std();
			CHECK(m_noiseStd >= 0) << "noise_std should be >= 0";
		}

		if (params.has_noise_mean())
		{
			m_noiseMean = params.noise_mean();
		}

		if (params.has_noise_std_small())
		{
			m_noiseStdSmall = params.noise_std_small();
			CHECK(m_noiseStdSmall >= 0) << "nose_std_small should be >= 0";
		}

		if (params.has_rotate_min_angle())
		{
			m_rotateMinAngle = params.rotate_min_angle();
		}

		if (params.has_rotate_max_angle())
		{
			m_rotateMaxAngle = params.rotate_max_angle();
		}

		CHECK(m_rotateMinAngle <= m_rotateMaxAngle)
			<< "rotate_min_angle should be < than rotate_max_angle";

		if (params.has_rotate_fill_value())
		{
			m_rotateFillValue = params.rotate_fill_value();
		}

		if (params.has_passthrough_probability())
		{
			m_passthroughProbability = params.passthrough_probability();
		}

		CHECK(m_passthroughProbability >= 0 && m_passthroughProbability <= 1)
			<< "passthrough_probability must be >= 0 and <= 1";

		if (params.has_maxout_passthrough_probability_iteration())
		{
			m_maxoutPassthroughProbabilityIteration =
					int(params.maxout_passthrough_probability_iteration());
		}

		CHECK(m_maxoutPassthroughProbabilityIteration >= 0)
					<< "maxout_passthrough_probability_iteration must be >= 0";

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

		const int numberOfBlobs = bottomBlob->num();
		const int width  = bottomBlob->width();
		const int height = bottomBlob->height();
		const int channels = bottomBlob->channels();

		cv::Size size(width, height);
		cv::Point2f center(width / 2.0, height / 2.0);

		//--------- passthrough distribution
		std::vector<int> passthroughs(numberOfBlobs, 0);

		caffe_rng_bernoulli<float>(
				numberOfBlobs,
				m_passthroughProbability,
				passthroughs.data());

		//--------- angle distribution
		std::vector<float> angles(numberOfBlobs, 0);

		caffe_rng_uniform<float>(
				numberOfBlobs,
				m_rotateMinAngle,
				m_rotateMaxAngle,
				angles.data());

		//--------- rescale distribution
		std::vector<int> rescales(numberOfBlobs, 0);

		caffe_rng_bernoulli<float>(
				numberOfBlobs,
				m_rescaleProbability,
				rescales.data());

		//--------- multiplier distribution
		std::vector<Dtype> multipliers(numberOfBlobs, Dtype(1));

		if ((m_constantMultiplierMean != 0.0f || m_constantMultiplierMean != 1.0f) &&
			m_constantMultiplierStd != 0.0f)
		{
			caffe_rng_gaussian<Dtype>(
					numberOfBlobs,
					m_constantMultiplierMean,
					m_constantMultiplierStd,
					multipliers.data());
		}

		//--------- color multiplier distribution
		std::vector<Dtype> multipliersColor(numberOfBlobs * channels, Dtype(1));

		if ((m_constantMultiplierColorMean != 0.0f || m_constantMultiplierColorMean != 1.0f) &&
			m_constantMultiplierColorStd != 0.0f)
		{
			caffe_rng_gaussian<Dtype>(
					numberOfBlobs * channels,
					m_constantMultiplierColorMean,
					m_constantMultiplierColorStd,
					multipliersColor.data());
		}

		//--------- scale distribution
		std::vector<float> scales(numberOfBlobs, 1.0f);

		if (m_scaleMean != 0.0f &&
			m_scaleStd != 0.0f)
		{
			caffe_rng_gaussian<float>(
					numberOfBlobs,
					m_scaleMean,
					m_scaleStd,
					scales.data());
		}

		//--------- iterate over images in blob
		#pragma omp parallel for
		for (int n = 0; n < numberOfBlobs; ++n)
		{
			const bool passthrough = passthroughs[n] > 0;

			if (passthrough == true)
			{
				if (bottomBlob != topBlob)
				{
					const int bottomOffset = bottomBlob->offset(n,0,0,0);
					const int topOffset = topBlob->offset(n,0,0,0);
					caffe_copy(
							width * height * channels,
							bottomData + bottomOffset,
							topData + topOffset);
				}
			}
			else
			{
				const float angle = angles[n];
				const float scale = scales[n];
				const bool rescale = rescales[n] > 0;
				const Dtype multiplier = std::max(Dtype(0), multipliers[n]);

				cv::Mat noiseSlice;
				cv::Mat noiseSmallSlice;
				cv::Mat perPixelMultiplierSlice;
				const cv::Mat rotationMatrix =
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
				if ((m_perPixelMultiplierMean != Dtype(0) && m_perPixelMultiplierMean != Dtype(1)) &&
					m_perPixelMultiplierStd != Dtype(0))
				{
					perPixelMultiplierSlice = cv::Mat(height, width, cv::DataType<Dtype>::type);
					caffe_rng_gaussian<Dtype>(
							width * height,
							Dtype(m_perPixelMultiplierMean),
							Dtype(m_perPixelMultiplierStd),
							(Dtype*)perPixelMultiplierSlice.data);
				}

				//----- iterate over channels
				for (int c = 0; c < channels; ++c)
				{
					const Dtype multiplierColor = multipliersColor[n * channels + c];
					const Dtype multiplierTotal = multiplier * multiplierColor;
					const int offset = bottomBlob->offset(n, c, 0, 0);

					cv::Mat bottomSlice(
							height,
							width,
							cv::DataType<Dtype>::type,
							(Dtype*)(bottomData + offset));

					//----- rotate
					if (angle == 0 &&
						scale == 0)
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
					if (multiplierTotal != Dtype(0) &&
						multiplierTotal != Dtype(1))
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
								topData + offset));
				}
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
		if (!propagate_down[0])
		{
			return;
		}

		if (top[0] == bottom[0])
		{
			return;
		}

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_copy(top[0]->count(), top_diff, bottom_diff);
	}

	//==================================================================
	
	INSTANTIATE_CLASS(VisionTransformationLayer);
	REGISTER_LAYER_CLASS(VisionTransformation);
};
