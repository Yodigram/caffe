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

	/*
		layer {
		  name: "sliding_histogram"
		  type: "SlidingHistogram"
		  bottom: "data"
		  top: "data"
		  sliding_histogram_param {
			kernel_size : 5
			stride : 3
			bins : 32
			min : -1
			max : 1
			flatten : true
		  }
		}
	*/

	//==================================================================

	template <typename Dtype>
	void SlidingHistogramLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top)
	{
		SlidingHistogramParameter params =
				this->layer_param_.sliding_histogram_param();
		//----------------------------------------
	    CHECK(!params.has_kernel_size() !=
	      !(params.has_kernel_h() && params.has_kernel_w()))
	      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
	    CHECK(params.has_kernel_size() ||
	      (params.has_kernel_h() && params.has_kernel_w()))
	      << "For non-square filters both kernel_h and kernel_w are required.";

	    if (params.has_kernel_size())
	    {
	    	m_kernel_h_ = m_kernel_w_ = params.kernel_size();
	    }
	    else
	    {
	    	m_kernel_h_ = params.kernel_h();
	    	m_kernel_w_ = params.kernel_w();
	    }

	    CHECK_GT(m_kernel_h_, 0) << "Filter dimensions cannot be zero.";
	    CHECK_GT(m_kernel_w_, 0) << "Filter dimensions cannot be zero.";
	    //----------------------------------------
	    if (params.has_min())
	    {
	    	m_min = Dtype(params.min());
	    }

	    if (params.has_max())
	    {
	    	m_max = Dtype(params.max());
	    }

	    CHECK_GT(m_max, m_min) << "max must be > min.";
	    //----------------------------------------
	    if (!params.has_stride_h())
	    {
	    	m_stride_h_ = m_stride_w_ = params.stride();
	    }
	    else
	    {
	    	m_stride_h_ = params.stride_h();
	    	m_stride_w_ = params.stride_w();
	    }

	    CHECK_GT(m_stride_h_, 0) << "stride_h must be > 0";
	    CHECK_GT(m_stride_w_, 0) << "stride_w must be > 0";
	    //----------------------------------------
	    if (params.has_bins())
	    {
	    	m_bins = int(params.bins());
	    }

	    CHECK_GT(m_bins, 1) << "bins must be > 1";
	    //----------------------------------------
	    if (params.has_diffuse())
	    {
	    	m_diffuseValue = params.diffuse();
	    }
	    //----------------------------------------
	    m_multiplier = (Dtype(m_bins) / (m_max - m_min));
	    //----------------------------------------
	    if (params.has_flatten())
	    {
	    	m_flatten = params.flatten();
	    }
	    //----------------------------------------
	}
	//==================================================================

	template <typename Dtype>
	void SlidingHistogramLayer<Dtype>::Reshape(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes())
				<< "Input must have 4 axes, "
				<< "corresponding to (num, channels, height, width)";
		m_channels_ = bottom[0]->channels();
		m_height_ = bottom[0]->height();
		m_width_ = bottom[0]->width();
		m_result_height_ = static_cast<int>(ceil(static_cast<float>(
				m_height_ - m_kernel_h_) / m_stride_h_)) + 1;
		m_result_width_ = static_cast<int>(ceil(static_cast<float>(
				m_width_ - m_kernel_w_) / m_stride_w_)) + 1;
		m_top_channels_ =
				(m_flatten == true) ? m_bins : m_channels_ * m_bins;

		top[0]->Reshape(
				bottom[0]->num(),
				m_top_channels_,
				m_result_height_,
				m_result_width_);

	    m_idx_.Reshape(
				1,
				1,
				m_result_height_,
				m_result_width_);

	    m_distance.Reshape(
				1,
				1,
				m_result_height_,
				m_result_width_);

	    //-------------------------------------------------
	    // set distance and index precalculated blobs
	    //-------------------------------------------------

		// zero out top and indices
		int* index_data = m_idx_.mutable_cpu_data();
		int* distance_data = m_distance.mutable_cpu_data();

		caffe_set(m_idx_.count(), int(-999999), index_data);
		caffe_set(m_distance.count(), int(m_kernel_w_ + m_kernel_h_), distance_data);

		for (int ph = 0; ph < m_result_height_; ++ph)
		{
			const int hstart = std::max(ph * m_stride_h_, 0);
			const int hend = std::min(hstart + m_kernel_h_, m_height_);
			const int center_h = (hstart + hend) / 2;

			for (int pw = 0; pw < m_result_width_; ++pw)
			{
				const int wstart = std::max(pw * m_stride_w_, 0);
				const int wend = std::min(wstart + m_kernel_w_, m_width_);
				const int center_w = (wstart + wend) / 2;

				//--------------------
				// run window and fill histogram
				for (int h = hstart; h < hend; ++h)
				{
					const int distance_h = std::abs(h - center_h);

					for (int w = wstart; w < wend; ++w)
					{
						const int distance_w = std::abs(w - center_w);
						const int distance = distance_h + distance_w;

						const int top_index =
								m_idx_.offset_unchecked(
										0,
										0,
										ph,
										pw);

						if (distance_data[top_index] > distance)
						{
							distance_data[top_index] = distance;
							index_data[top_index] = h * m_width_ + w;
						}
					}
				}
			}
		}
	}

	//==================================================================

	template <typename Dtype>
	void SlidingHistogramLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top)
	{
		Blob<Dtype>* bottom_blob = bottom[0];
		const Dtype* bottom_data = bottom_blob->cpu_data();
		Blob<Dtype>* top_blob = top[0];
		Dtype* top_data = top_blob->mutable_cpu_data();
		const int num = bottom_blob->num();

		// zero out top
		caffe_set(top_blob->count(), m_diffuseValue, top_data);

		#pragma omp parallel for
		for (int n = 0; n < num; ++n)
		{
			for (int c = 0; c < m_channels_; ++c)
			{
				const int offset = bottom_blob->offset_unchecked(n, c, 0, 0);
				const int channelOffset = (m_flatten == true) ? 0 : (c * m_bins);

				for (int ph = 0; ph < m_result_height_; ++ph)
				{
					const int hstart = std::max(ph * m_stride_h_, 0);
					const int hend = std::min(hstart + m_kernel_h_, m_height_);

					for (int pw = 0; pw < m_result_width_; ++pw)
					{
						const int wstart = std::max(pw * m_stride_w_, 0);
						const int wend = std::min(wstart + m_kernel_w_, m_width_);

						//--------------------
						// add this value to bins so we dont need 2 passes to normalize
						const Dtype value =
								(Dtype(1) - Dtype(m_bins) * m_diffuseValue) /
									Dtype((hend - hstart) * (wstart - wend));

						//--------------------
						// run window and fill histogram
						for (int h = hstart; h < hend; ++h)
						{
							for (int w = wstart; w < wend; ++w)
							{
								const int bottom_index = offset + h * m_width_ + w;
								const int bin = bin_of_value(bottom_data[bottom_index]);
								const int top_index =
										top_blob->offset_unchecked(
												n,
												channelOffset + bin,
												ph,
												pw);
								top_data[top_index] += value;
							}
						}
						//--------------------
					}
				}
			}
		}
	}

	//==================================================================

	template <typename Dtype>
	void SlidingHistogramLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom)
	{
		if (!propagate_down[0])
		{
			return;
		}

		Blob<Dtype>* bottom_blob = bottom[0];
		Dtype* bottom_diff = bottom_blob->mutable_cpu_diff();
		Blob<Dtype>* top_blob = top[0];
		const Dtype* top_diff = top_blob->cpu_diff();
		const int* index_data = m_idx_.cpu_data();
		const int num = bottom_blob->num();

		// zero out diff
		caffe_set(bottom_blob->count(), Dtype(0), bottom_diff);

		if (m_flatten == true)
		{
			#pragma omp parallel for
			for (int n = 0; n < num; ++n)
			{
				for (int c = 0; c < m_top_channels_; ++c)
				{
					const int top_offset =
							top_blob->offset_unchecked(
									n,
									c,
									0,
									0);

					for (int h = 0; h < m_result_height_; ++h)
					{
						for (int w = 0; w < m_result_width_; ++w)
						{
							const int tmpIndex = h * m_result_width_ + w;
							const int top_index = top_offset + tmpIndex;
							Dtype value = top_diff[top_index] / Dtype(m_bins);

							for (int i = 0; i < m_bins; ++i)
							{
								const int bottom_index =
										bottom_blob->offset_unchecked(n,i,0,0) +
										index_data[tmpIndex];
								bottom_diff[bottom_index] += value;
							}
						}
					}
				}
			}
		}
		else
		{
			#pragma omp parallel for
			for (int n = 0; n < num; ++n)
			{
				for (int c = 0; c < m_top_channels_; ++c)
				{
					const int top_offset =
							top_blob->offset_unchecked(
									n,
									c,
									0,
									0);
					const int bottom_offset =
							bottom_blob->offset_unchecked(
									n,
									c / m_bins,
									0,
									0);

					for (int h = 0; h < m_result_height_; ++h)
					{
						for (int w = 0; w < m_result_width_; ++w)
						{
							const int tmpIndex = h * m_result_width_ + w;
							const int top_index = top_offset + tmpIndex;
							const int bottom_index = bottom_offset + index_data[tmpIndex];
							bottom_diff[bottom_index] += top_diff[top_index];
						}
					}
				}
			}
		}
	}

	//==================================================================

	INSTANTIATE_CLASS(SlidingHistogramLayer);
	REGISTER_LAYER_CLASS(SlidingHistogram);

	//==================================================================
};
