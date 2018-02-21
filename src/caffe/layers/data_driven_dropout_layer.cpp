#include <vector>
#include <omp.h>
#include <cmath>
#include <caffe/caffe.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/heap/pairing_heap.hpp>
#include <boost/heap/priority_queue.hpp>

#include "caffe/blob.hpp"
#include "caffe/yodi_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	//=========================================================
	/* Example prototxt
	 *
		layer {
		  name: "ddd_0"
		  type: "DataDrivenDropout"
		  bottom: "data"
		  top: "data"
		  data_driven_dropout_param {
			topk: 0.1
			filter_method: FULL
			absolute_value: true
			passthrough_probability: 0.0
		  }
		  include {
			phase: TRAIN
		  }
		}
	*/

	//=========================================================

	template<typename Dtype>
	void DataDrivenDropoutLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype> *>& bottom,
		const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);

	    CHECK(this->layer_param_.has_data_driven_dropout_param() == true)
			<< "Data driven dropout parameters are missing";

	    DataDrivenDropoutParameter params = this->layer_param_.data_driven_dropout_param();

		if (params.has_topk() == true)
		{
			m_topk = params.topk();
			CHECK(m_topk > 0) << "m_topk should be > 0";
			CHECK(m_topk <= 1) << "m_topk should be <= 1";
			m_scale *= Dtype(1) / Dtype(m_topk);
		}

		if (params.has_cutoff_value() == true)
		{
			m_cutoff_value = params.cutoff_value();
		}

		if (params.has_absolute_value() == true)
		{
			m_absoluteValue = params.absolute_value();
		}

		if (params.has_filter_method() == true)
		{
			m_filter_method = params.filter_method();
		}

		if (params.has_passthrough_probability())
		{
			m_passthrough_probability = params.passthrough_probability();
			CHECK(m_passthrough_probability >= 0 && m_passthrough_probability <= 1)
				<< "m_passthrough_probability must be >= 0 and <= 1";
		}
	}

	//=========================================================

	template<typename Dtype>
	void DataDrivenDropoutLayer<Dtype>::Reshape(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		NeuronLayer<Dtype>::Reshape(bottom, top);

		m_mask.Reshape(bottom[0]->shape());
	}

	//=========================================================

	template<typename Dtype>
	void DataDrivenDropoutLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype> *>& bottom,
			const vector<Blob<Dtype> *>& top)
	{
		Blob<Dtype>* top_blob = top[0];
		Blob<Dtype>* bottom_blob = bottom[0];
		const Dtype* bottom_data = bottom_blob->cpu_data();
		Dtype* top_data = top_blob->mutable_cpu_data();
		Dtype* mask_data = m_mask.mutable_cpu_data();

		const int no_blobs = bottom_blob->num();
		const int width  = bottom_blob->width();
		const int height = bottom_blob->height();
		const int channels = bottom_blob->channels();

		//--------- passthrough distribution
		std::vector<int> passthroughs;
		const bool calculate_passthrough =
				this->phase_ == TRAIN &&
		    	m_passthrough_probability > 0.0f;

	    if (calculate_passthrough == true)
	    {
	    	passthroughs.resize(no_blobs);

			caffe_rng_bernoulli<float>(
					no_blobs,
					m_passthrough_probability,
					passthroughs.data());
	    }

		//--------- iterate over blobs
		#pragma omp parallel for
		for (int n = 0; n < no_blobs; ++n)
		{
			const bool absolute = m_absoluteValue;

			if (calculate_passthrough == true)
			{
				const int no_elements = width * height * channels;
				const int mask_offset = m_mask.offset(n, 0, 0, 0);
				const int top_offset = top_blob->offset(n, 0, 0, 0);
				const int bottom_offset = bottom_blob->offset(n, 0, 0, 0);

				if (passthroughs[n] == true)
				{
					caffe_copy(
							no_elements,
							bottom_data + bottom_offset,
							top_data + top_offset);
					caffe_set((int)no_elements, Dtype(1), mask_data + mask_offset);
					continue;
				}
			}

			//----------------------------- CHANNELWISE
			if (m_filter_method == DataDrivenDropoutParameter_FilterMethod_CHANNELWISE)
			{
				// go through the channels and for each one keep only top_k % of the top
				for (int c = 0; c < channels; ++c)
				{
					const int no_elements = width * height;
					const int mask_offset = m_mask.offset(n, c, 0, 0);
					const int top_offset = top_blob->offset(n, c, 0, 0);
					const int bottom_offset = bottom_blob->offset(n, c, 0, 0);
					const int no_top_k_elements = std::max(1, int(std::round(float(no_elements) * m_topk)));

					// find top k-th element of blob
					Dtype kth_top_element = 0;
					{
						boost::heap::priority_queue<
							Dtype,
							boost::heap::compare< std::greater< Dtype > > > top_k_queue;

						for (int i = 0; i < no_elements; ++i)
						{
							const Dtype b = bottom_data[bottom_offset + i];
							const Dtype d = absolute ? std::abs(b) : b;
							const int size = top_k_queue.size();

							if (size < no_top_k_elements)
							{
								top_k_queue.push(d);
							}
							else if (d >= top_k_queue.top())
							{
								top_k_queue.pop();
								top_k_queue.push(d);
							}
						}

						kth_top_element = top_k_queue.top();
					}

					// mask all elements <= kth_top_element
					for (int i = 0; i < no_elements; ++i)
					{
						const Dtype b = bottom_data[bottom_offset + i];
						const Dtype d = absolute ? std::abs(b) : b;
						const bool pass = d >= kth_top_element;
						const Dtype m = pass ? m_scale : m_cutoff_value;
						mask_data[mask_offset + i] = m;
						top_data[top_offset + i] = Dtype(m * d);
					}
				}
			}
			//----------------------------- CHANNELWIDE
			else if (m_filter_method == DataDrivenDropoutParameter_FilterMethod_CHANNELWIDE)
			{
				// go through the channels, count each one's value and keep only the top_k% channels
				const int no_elements = width * height;
				std::vector<Dtype> values(channels, 0);
				const int no_top_k_elements = std::max(1, int(std::round(float(channels) * m_topk)));

				for (int c = 0; c < channels; ++c)
				{
					const int bottom_offset = bottom_blob->offset(n, c, 0, 0);

					for (int i = 0; i < no_elements; ++i)
					{
						const Dtype b = bottom_data[bottom_offset + i];
						const Dtype d = absolute ? std::abs(b) : b;
						values[c] += d;
					}
				}

				Dtype kth_top_element = 0;
				{
					boost::heap::priority_queue<
						Dtype,
						boost::heap::compare< std::greater< Dtype > > > top_k_queue;

					for (int c = 0; c < channels; ++c)
					{
						const Dtype d = values[c];
						const int size = top_k_queue.size();

						if (size < no_top_k_elements)
						{
							top_k_queue.push(d);
						}
						else if (d >= top_k_queue.top())
						{
							top_k_queue.pop();
							top_k_queue.push(d);
						}
					}

					kth_top_element = top_k_queue.top();
				}


				for (int c = 0; c < channels; ++c)
				{
					const bool pass = values[c] >= kth_top_element;
					const Dtype m = pass ? m_scale : m_cutoff_value;
					const int mask_offset = m_mask.offset(n, c, 0, 0);
					const int top_offset = top_blob->offset(n, c, 0, 0);
					const int bottom_offset = bottom_blob->offset(n, c, 0, 0);

					for (int i = 0; i < no_elements; ++i)
					{
						mask_data[mask_offset + i] = m;
						top_data[top_offset + i] = m * bottom_data[bottom_offset + i];
					}
				}
			}
			//----------------------------- FULL
			else if (m_filter_method == DataDrivenDropoutParameter_FilterMethod_FULL)
			{
				// go through all the values and one keep only top_k % of the top
				const int no_elements = width * height * channels;
				const int mask_offset = m_mask.offset(n, 0, 0, 0);
				const int top_offset = top_blob->offset(n, 0, 0, 0);
				const int bottom_offset = bottom_blob->offset(n, 0, 0, 0);
				const int no_top_k_elements = std::max(1, int(std::round(float(no_elements) * m_topk)));

				// find top k-th element of blob
				Dtype kth_top_element = 0;
				{
					boost::heap::priority_queue<
						Dtype,
						boost::heap::compare< std::greater< Dtype > > > top_k_queue;

					for (int i = 0; i < no_elements; ++i)
					{
						const Dtype b = bottom_data[bottom_offset + i];
						const Dtype d = absolute ? std::abs(b) : b;
						const int size = top_k_queue.size();

						if (size < no_top_k_elements)
						{
							top_k_queue.push(d);
						}
						else if (d >= top_k_queue.top())
						{
							top_k_queue.pop();
							top_k_queue.push(d);
						}
					}

					kth_top_element = top_k_queue.top();
				}

				// mask all elements <= kth_top_element
				for (int i = 0; i < no_elements; ++i)
				{
					const Dtype b = bottom_data[bottom_offset + i];
					const Dtype d = absolute ? std::abs(b) : b;
					const bool pass = d >= kth_top_element;
					const Dtype m = pass ? m_scale : m_cutoff_value;
					mask_data[mask_offset + i] = m;
					top_data[top_offset + i] = Dtype(m * d);
				}
			}
		}
	}

	//==================================================================

	template<typename Dtype>
	void DataDrivenDropoutLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype> *>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom)
	{
		if (propagate_down[0] == false)
		{
			return;
		}

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	    if (this->phase_ == TRAIN)
	    {
	    	const Dtype* mask = m_mask.cpu_data();
	    	const int count = bottom[0]->count();

	    	#pragma omp parallel for
	    	for (int i = 0; i < count; ++i)
	    	{
	    		bottom_diff[i] = Dtype(top_diff[i] * mask[i] * m_scale);
	    	}
	    }
	    else
	    {
	    	if (bottom[0] != top[0])
	    	{
	    		caffe_copy(top[0]->count(), top_diff, bottom_diff);
	    	}
	    }
	}

	//==================================================================

	INSTANTIATE_CLASS(DataDrivenDropoutLayer);
	REGISTER_LAYER_CLASS(DataDrivenDropout);

	//=========================================================
}
